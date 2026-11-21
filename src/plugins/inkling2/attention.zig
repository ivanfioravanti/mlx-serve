//! Inkling 2's attention (mlx-lm `Attention`, `SlidingAttention`, `FullAttention`): no RoPE. q is scaled and
//! L2-clipped per head, k clipped, v RMS-normed with no gain; a hidden-conditioned relative bias and one sink logit
//! per query join the softmax; the output passes a sigmoid gate. k and v pass a short conv first, static on sliding
//! layers and hidden-conditioned on global ones, which also scale q and the bias by log(position) past the floor.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const band = @import("../../sdk/bases/blocks/sliding.zig");
const sdpaSinks = @import("../../sdk/bases/blocks/decoder_attention.zig").sdpaSinks;
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const blocks = @import("blocks.zig");
const rel_attn = @import("rel_attn.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Ctx = block.Ctx;

pub const Weights = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    rel: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    /// `[heads, head_dim]`.
    q_clip: Array,
    /// `[heads, d_rel]`.
    rel_clip: Array,
    /// `[d_rel, extent]`.
    rel_proj: Array,
    conv: union(enum) {
        /// Depthwise taps `[C, K, 1]`.
        static: struct { k: Array, v: Array },
        /// Per-head taps `[K, C]` plus the layer input's projection to `heads * K`.
        dynamic: struct { k: Array, v: Array, k_proj: Linear, v_proj: Linear },
    },
    /// q, rel, gate and sink (one input).
    join_q: Joined = .{},
    /// A sliding layer's k and v (one input unless a loop message tells them apart); a global layer's k with its conv
    /// projection, and v with its.
    join_kv: Joined = .{},
    join_k: Joined = .{},
    join_v: Joined = .{},
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    d_rel: c_int,
    /// 0 = a global layer.
    window: u32,
    log_floor: u32,
    log_alpha: f32,
    qk_clip: f32,
    rel_clip_at: f32,
    eps: f32,
};

pub const Out = struct { out: Array, k: blocks.Conv, v: blocks.Conv };

const Joined = @import("../../sdk/bases/blocks/linear.zig").Joined;

/// The first `n` parts of `j`'s one matmul over `x`.
fn split(j: *const Joined, g: *Graph, x: Array, comptime n: usize) ![n]Array {
    const y = try j.lin.forward(g, x);
    var out: [n]Array = undefined;
    for (&out, 0..) |*o, i| o.* = try j.part(g, y, i);
    return out;
}

/// The layer's attention over its normed q, k and v inputs `[1, L, hidden]`; `k_prev`/`v_prev` are the conv windows.
pub fn forward(w: *const Weights, c: *Ctx, qs: Array, ks: Array, vs: Array, k_prev: ?Array, v_prev: ?Array) !Out {
    const g = c.g;
    const l = c.rows;
    const hd = w.head_dim;
    const offset: c_int = @intCast(c.offset());
    const total = offset + l;
    const kc, const vc = switch (w.conv) {
        .static => |s| blk: {
            var kv_lin: [2]Array = undefined;
            if (w.join_kv.built() and ks.ctx == vs.ctx) {
                kv_lin = try split(&w.join_kv, g, ks, 2);
            } else kv_lin = .{ try w.k.forward(g, ks), try w.v.forward(g, vs) };
            break :blk .{ try blocks.sconv(g, kv_lin[0], s.k, k_prev), try blocks.sconv(g, kv_lin[1], s.v, v_prev) };
        },
        .dynamic => |d| blk: {
            const kk = if (w.join_k.built()) try split(&w.join_k, g, ks, 2) else [2]Array{ try w.k.forward(g, ks), try d.k_proj.forward(g, ks) };
            const vv = if (w.join_v.built()) try split(&w.join_v, g, vs, 2) else [2]Array{ try w.v.forward(g, vs), try d.v_proj.forward(g, vs) };
            break :blk .{
                try blocks.dynamicSconv(g, kk[0], kk[1], d.k, w.kv_heads, k_prev),
                try blocks.dynamicSconv(g, vv[0], vv[1], d.v, w.kv_heads, v_prev),
            };
        },
    };
    const qo = if (w.join_q.built()) try split(&w.join_q, g, qs, 4) else [4]Array{ try w.q.forward(g, qs), try w.rel.forward(g, qs), try w.gate.forward(g, qs), try w.sink.forward(g, qs) };
    const gate = qo[2];
    const sinks = qo[3];

    var q = try blocks.l2Clip(g, try g.reshape(qo[0], &.{ 1, l, w.heads, hd }), w.q_clip, w.qk_clip);
    const k = try blocks.l2Clip(g, try g.reshape(kc.out, &.{ 1, l, w.kv_heads, hd }), null, w.qk_clip);
    const v4 = try g.reshape(vc.out, &.{ 1, l, w.kv_heads, hd });
    const v = (try blocks.rmsCast(g, v4, null, w.eps, ops.dtype(v4), false)).out;
    var rel = try blocks.l2Clip(g, try g.reshape(qo[1], &.{ 1, l, w.heads, w.d_rel }), w.rel_clip, w.rel_clip_at);
    if (w.window == 0 and total > @as(c_int, @intCast(w.log_floor))) {
        const tau = try logScale(g, offset, l, w.log_floor, w.log_alpha);
        q = try g.astype(try g.mul(try blocks.f32Of(g, q), tau), ops.dtype(q));
        rel = try g.astype(try g.mul(try blocks.f32Of(g, rel), tau), ops.dtype(rel));
    }
    const bias = try g.matmul(rel, try g.astype(w.rel_proj, ops.dtype(rel)));

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const sv = band.view(w.window, hd, total, l);
    var view = try kv.append(c, try g.transpose(k, &perm), try g.transpose(v, &perm), if (w.window > 0) sv.span else 0);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const out = try attend(g, .{
        .q = try g.transpose(q, &perm),
        .k = dv.k,
        .v = dv.v,
        .rel = bias,
        .sinks = sinks,
        .key_start = try g.zeros(&.{1}, .int32),
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        .window = w.window,
    });
    const gated = try @import("compiled.zig").sigmoidGate(g, try g.reshape(out, &.{ 1, l, w.heads * hd }), gate);
    return .{ .out = try w.o.forward(g, gated), .k = kc, .v = vc };
}

/// `1 + alpha * log(max((position + 1) / floor, 1))` per query row, `[1, L, 1, 1]` f32.
fn logScale(g: *Graph, offset: c_int, l: c_int, floor: u32, alpha: f32) !Array {
    const n = try g.arange(@floatFromInt(offset + 1), @floatFromInt(offset + l + 1), 1, .float32);
    const ratio = try g.maximum(try g.mulScalar(n, 1.0 / @as(f32, @floatFromInt(floor))), try g.scalar(1, .float32));
    return g.reshape(try g.addScalar(try g.mulScalar(try g.log(ratio), alpha), 1), &.{ 1, l, 1, 1 });
}

/// Keys from which a window is read over key chunks; under it, row by row is as quick. A single row stays on
/// the fused attention at every length (over 65k keys it is half again as fast as the chunked kernel).
const SPLIT_KEYS = 4096;

/// `[1, L, Hq, D]`: one decode row through MLX's fused attention (its sinks are one row's), a chunk through the kernels.
pub fn attend(g: *Graph, a: rel_attn.Args) !Array {
    const qs = ops.shape(a.q);
    if (qs[2] != 1) {
        if (qs[2] > rel_attn.MIN_ROWS) return rel_attn.attend(g, a);
        // A window under the kernel's floor (a speculative round's verify rows): each row as a decode row reads
        // it, over its own keys. The op chain scores every key for every row and rounds differently from decode.
        const total = ops.shape(a.k)[2];
        if (total >= SPLIT_KEYS) if (try rel_attn.split(g, a)) |out| return out;
        var outs: [rel_attn.MIN_ROWS]Array = undefined;
        for (0..@intCast(qs[2])) |i| {
            const at: c_int = @intCast(i);
            const end = total - qs[2] + at + 1;
            const from: c_int = if (a.window > 0) @max(0, end - @as(c_int, @intCast(a.window))) else 0;
            outs[i] = try attend(g, .{
                .q = try g.sliceAxis(a.q, 2, at, at + 1),
                .k = try g.sliceAxis(a.k, 2, from, end),
                .v = try g.sliceAxis(a.v, 2, from, end),
                .rel = try g.sliceAxis(a.rel, 1, at, at + 1),
                .sinks = try g.sliceAxis(a.sinks, 1, at, at + 1),
                .key_start = a.key_start,
                .scale = a.scale,
                .window = a.window,
            });
        }
        return g.concat(outs[0..@intCast(qs[2])], 1);
    }
    const hq = qs[1];
    const s = ops.shape(a.k)[2];
    const r = ops.shape(a.rel)[3];
    const dt = ops.dtype(a.q);
    const mask = try rowBias(g, try g.reshape(a.rel, &.{ hq, r }), s, a.window);
    const y = try sdpaSinks(g, a.q, a.k, a.v, a.scale, "array", try g.reshape(try g.astype(mask, dt), &.{ 1, hq, 1, s }), try g.astype(try g.reshape(a.sinks, &.{hq}), dt), false);
    return g.transpose(y, &.{ 0, 2, 1, 3 });
}

/// One row's additive mask `[hq, s]` from its bias `[hq, r]`: key j sits `s - 1 - j` behind the row, so the mask
/// is the bias read backwards, zero past the table and -inf past the window.
fn rowBias(g: *Graph, rel: Array, s: c_int, window: u32) !Array {
    const hq = ops.shape(rel)[0];
    const r = ops.shape(rel)[1];
    if (window == 0 or s <= @as(c_int, @intCast(window))) {
        // No key is past the window: the reversed bias is a view, and only an open layer pads it.
        const n = @min(r, s);
        const near = try g.call(mlx.mlx_slice, .{ rel, &[_]c_int{ 0, n - 1 }, @as(usize, 2), &[_]c_int{ hq, -r - 1 }, @as(usize, 2), &[_]c_int{ 1, -1 }, @as(usize, 2) });
        return if (s > r) g.concat(&.{ try g.zeros(&.{ hq, s - r }, ops.dtype(rel)), near }, 1) else near;
    }
    const dist = try g.arange(@floatFromInt(s - 1), -1, -1, .int32);
    const int = struct {
        fn of(gr: *Graph, v: c_int) !Array {
            return gr.adopt(mlx.mlx_array_new_int(v));
        }
    }.of;
    const near = try g.take(rel, try g.minimum(dist, try int(g, r - 1)), 1);
    const mask = try g.where(try g.less(dist, try int(g, r)), near, try g.scalarLike(near, 0));
    return g.where(try g.less(dist, try int(g, @intCast(window))), mask, try g.scalarLike(near, -std.math.inf(f32)));
}

// ── tests ──

const testing = std.testing;

// Bar: the fused one-row arm equals the reference chain's last row.
test "inkling2 attention: a decode row and a short window equal the chunk's rows" {
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const hq = 4;
    const hkv = 2;
    const d = 8;
    const s = 9;
    const l = 5;
    const r = 4;
    var prng = std.Random.DefaultPrng.init(11);
    const fill = struct {
        fn f(gr: *Graph, p: *std.Random.DefaultPrng, comptime n: usize, sh: []const c_int) !Array {
            var buf: [n]f32 = undefined;
            for (&buf) |*x| x.* = p.random().floatNorm(f32);
            return gr.fromSlice(f32, &buf, sh);
        }
    }.f;
    const q = try fill(&g, &prng, hq * l * d, &.{ 1, hq, l, d });
    const k = try fill(&g, &prng, hkv * s * d, &.{ 1, hkv, s, d });
    const v = try fill(&g, &prng, hkv * s * d, &.{ 1, hkv, s, d });
    const rel = try fill(&g, &prng, l * hq * r, &.{ 1, l, hq, r });
    const sinks = try fill(&g, &prng, l * hq, &.{ 1, l, hq });
    for ([_]u32{ 0, 3 }) |window| {
        const start = try g.zeros(&.{1}, .int32);
        const whole = try rel_attn.reference(&g, .{ .q = q, .k = k, .v = v, .rel = rel, .sinks = sinks, .key_start = start, .scale = 0.35, .window = window });
        const one = try attend(&g, .{
            .q = try g.sliceAxis(q, 2, l - 1, l),
            .k = k,
            .v = v,
            .rel = try g.sliceAxis(rel, 1, l - 1, l),
            .sinks = try g.sliceAxis(sinks, 1, l - 1, l),
            .key_start = start,
            .scale = 0.35,
            .window = window,
        });
        const last = try g.sliceAxis(whole, 1, l - 1, l);
        try ops.evalMany(&.{ last, one });
        const diff = try g.max(try g.reshape(try g.abs(try g.sub(last, one)), &.{hq * d}), 0, false);
        try ops.eval(diff);
        try testing.expect(mlx.mlx_array_data_float32(diff).?[0] < 1e-5);
        // A short window is its rows, each read alone.
        const rows = try attend(&g, .{ .q = q, .k = k, .v = v, .rel = rel, .sinks = sinks, .key_start = start, .scale = 0.35, .window = window });
        const off = try g.max(try g.reshape(try g.abs(try g.sub(whole, rows)), &.{l * hq * d}), 0, false);
        try ops.eval(off);
        try testing.expect(mlx.mlx_array_data_float32(off).?[0] < 1e-5);
    }
}

test "inkling2 attention: a row's mask is its bias read backwards, zero past the table, -inf past the window" {
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const rel = try g.fromSlice(f32, &.{ 1, 2, 3, 4, 5, 10, 20, 30, 40, 50 }, &.{ 2, 5 });
    const inf = std.math.inf(f32);
    const cases = [_]struct { s: c_int, window: u32, want: []const f32 }{
        .{ .s = 3, .window = 0, .want = &.{ 3, 2, 1, 30, 20, 10 } },
        .{ .s = 5, .window = 8, .want = &.{ 5, 4, 3, 2, 1, 50, 40, 30, 20, 10 } },
        .{ .s = 7, .window = 0, .want = &.{ 0, 0, 5, 4, 3, 2, 1, 0, 0, 50, 40, 30, 20, 10 } },
        .{ .s = 4, .window = 2, .want = &.{ -inf, -inf, 2, 1, -inf, -inf, 20, 10 } },
    };
    for (cases) |cs| {
        const got = try g.contiguous(try rowBias(&g, rel, cs.s, cs.window));
        try ops.eval(got);
        try testing.expectEqualSlices(c_int, &.{ 2, cs.s }, ops.shape(got));
        try testing.expectEqualSlices(f32, cs.want, mlx.mlx_array_data_float32(got).?[0..cs.want.len]);
    }
}

// Bar: each part of the joined matmul is the separate projection's output, element for element, at a decode row and
// a 2048-row chunk, on 4-bit g64 affine weights of the pack's q, rel, gate and sink shapes.
test "inkling2 attention: the joined projections give each projection's own rows" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const loader = @import("../llama/load.zig");
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var o: loader.Owned = .{};
    defer o.release();
    const widths = [_]c_int{ 3072, 384, 3072, 24 };
    var lins: [4]Linear = undefined;
    for (&lins, widths, 0..) |*l, n, i| {
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, i)});
        const wf = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ n, 2304 }, @as(usize, 2), ops.Dtype.float32, @as(f32, 0), @as(f32, 0.05), key }), .bfloat16);
        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(&qv, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, g.s));
        var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        for (&parts, 0..) |*p, j| try mlx.check(mlx.mlx_vector_array_get(p, qv, j));
        try ops.evalMany(&parts);
        l.* = .{ .w = parts[0], .s = parts[1], .b = parts[2], .bits = 4, .group_size = 64, .mode = .affine };
    }
    const j = try loader.join(&o, &.{ &lins[0], &lins[1], &lins[2], &lins[3] }, g.s);
    try testing.expect(j.built());
    for (lins) |l| try o.linear(l);
    for ([_]c_int{ 1, 2048 }) |rows| {
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, @intCast(rows))});
        const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, rows, 2304 }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key }), .bfloat16);
        const y = try j.lin.forward(&g, x);
        for (lins, 0..) |l, i| {
            const a = try g.astype(try j.part(&g, y, i), .float32);
            const b = try g.astype(try l.forward(&g, x), .float32);
            const off = try g.abs(try g.sub(a, b));
            const differ = try g.call(mlx.mlx_sum, .{ try g.astype(try g.call(mlx.mlx_greater, .{ off, try g.scalar(0, .float32) }), .float32), false });
            // In bf16 ulps of the separate projection's value.
            const ulps = try g.call(mlx.mlx_max, .{ try g.div(off, try g.addScalar(try g.mulScalar(try g.abs(b), 1.0 / 128.0), 1e-30)), false });
            try ops.evalMany(&.{ differ, ulps });
            var d: f32 = 0;
            var u: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&d, differ));
            try mlx.check(mlx.mlx_array_item_float32(&u, ulps));
            if ((rows == 1 and d != 0) or u > 1.0) std.debug.print("joined part {d} at {d} rows: {d:.0} elements differ, worst {d:.2} ulp\n", .{ i, rows, d, u });
            // A decode row is the same kernel either way; at chunk width MLX may tile a narrow projection differently.
            if (rows == 1) try testing.expectEqual(@as(f32, 0), d) else try testing.expect(u <= 1.0);
        }
    }
}
