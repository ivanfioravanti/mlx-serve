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

/// The layer's attention over its normed q, k and v inputs `[1, L, hidden]`; `k_prev`/`v_prev` are the conv windows.
pub fn forward(w: *const Weights, c: *Ctx, qs: Array, ks: Array, vs: Array, k_prev: ?Array, v_prev: ?Array) !Out {
    const g = c.g;
    const l = c.rows;
    const hd = w.head_dim;
    const offset: c_int = @intCast(c.offset());
    const total = offset + l;
    const k_lin = try w.k.forward(g, ks);
    const v_lin = try w.v.forward(g, vs);
    const kc, const vc = switch (w.conv) {
        .static => |s| .{ try blocks.sconv(g, k_lin, s.k, k_prev), try blocks.sconv(g, v_lin, s.v, v_prev) },
        .dynamic => |d| .{
            try blocks.dynamicSconv(g, k_lin, try d.k_proj.forward(g, ks), d.k, w.kv_heads, k_prev),
            try blocks.dynamicSconv(g, v_lin, try d.v_proj.forward(g, vs), d.v, w.kv_heads, v_prev),
        },
    };
    const gate = try w.gate.forward(g, qs);
    const sinks = try w.sink.forward(g, qs);

    var q = try blocks.l2Clip(g, try g.reshape(try w.q.forward(g, qs), &.{ 1, l, w.heads, hd }), w.q_clip, w.qk_clip);
    const k = try blocks.l2Clip(g, try g.reshape(kc.out, &.{ 1, l, w.kv_heads, hd }), null, w.qk_clip);
    const v4 = try g.reshape(vc.out, &.{ 1, l, w.kv_heads, hd });
    const v = try g.astype(try blocks.rms(g, try blocks.f32Of(g, v4), w.eps), ops.dtype(v4));
    var rel = try blocks.l2Clip(g, try g.reshape(try w.rel.forward(g, qs), &.{ 1, l, w.heads, w.d_rel }), w.rel_clip, w.rel_clip_at);
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

/// `[1, L, Hq, D]`: one decode row through MLX's fused attention (its sinks are one row's), a chunk through the kernels.
pub fn attend(g: *Graph, a: rel_attn.Args) !Array {
    const qs = ops.shape(a.q);
    if (qs[2] != 1) return rel_attn.attend(g, a);
    const hq = qs[1];
    const s = ops.shape(a.k)[2];
    const r = ops.shape(a.rel)[3];
    const dt = ops.dtype(a.q);
    // Key j sits `s - 1 - j` behind the row.
    const dist = try g.arange(@floatFromInt(s - 1), -1, -1, .int32);
    const int = struct {
        fn of(gr: *Graph, v: c_int) !Array {
            return gr.adopt(mlx.mlx_array_new_int(v));
        }
    }.of;
    const at = try g.minimum(dist, try int(g, r - 1));
    const near = try g.take(try g.reshape(a.rel, &.{ hq, r }), at, 1);
    var mask = try g.where(try g.less(dist, try int(g, r)), near, try g.scalarLike(near, 0));
    if (a.window > 0) mask = try g.where(try g.less(dist, try int(g, @intCast(a.window))), mask, try g.scalarLike(near, -std.math.inf(f32)));
    const y = try sdpaSinks(g, a.q, a.k, a.v, a.scale, "array", try g.reshape(try g.astype(mask, dt), &.{ 1, hq, 1, s }), try g.astype(try g.reshape(a.sinks, &.{hq}), dt), false);
    return g.transpose(y, &.{ 0, 2, 1, 3 });
}

// ── tests ──

const testing = std.testing;

// Bar: the fused one-row arm equals the reference chain's last row.
test "inkling2 attention: the decode row equals the last row of the chunk" {
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
    }
}
