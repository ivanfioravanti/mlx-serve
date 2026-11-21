//! Inkling 2's routed MLP (mlx-lm `SparseMoeBlock`): an f32 sigmoid router with a selection-only bias, top-k
//! weights normalized and scaled, experts whose fused `gate_up_proj` rows alternate gate and up, a softcapped SwiGLU,
//! no shared expert. The bank's quant is its own (NVFP4 with a per-expert global scale, MXFP8, affine or dense).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const compiled = @import("compiled.zig");
const experts = @import("experts.zig");
const router_kernel = @import("../../sdk/blocks/moe/router_kernel.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub const Weights = struct {
    /// `[hidden, experts]` f32.
    router: Array,
    /// The selection bias `[experts]` f32.
    bias: Array,
    gate_up: Bank,
    down: Bank,
    /// Each expert's multiplier `[experts]` f32 on a bank's output (an NVFP4 global scale); null = none.
    gate_up_scale: Array = .{},
    down_scale: Array = .{},
    top_k: c_int,
    route_scale: f32,
    gate_softcap: f32,
    up_softcap: f32,
};

fn gather(g: *Graph, x: Array, bank: *const Bank, rhs: Array, sorted: bool) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, .{ .ctx = null }, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

const Routing = struct { idx: Array, weights: Array };

/// Each row's top-k experts by `sigmoid(logits) + bias` and their sigmoid scores, normalized and scaled: one
/// kernel, or the op chain where it declines.
fn route(w: *const Weights, g: *Graph, logits: Array) !Routing {
    if (try router_kernel.moeRouterTopK(g.s, logits, w.bias, w.top_k, .sigmoid_bias, true, w.route_scale, .float32, 0, 0, null)) |r| {
        return .{ .idx = try g.adopt(r.inds), .weights = try g.adopt(r.norm_scores) };
    }
    const scores = try g.sigmoid(logits);
    const picked = try g.argpartition(try g.neg(try g.add(scores, w.bias)), w.top_k - 1, -1);
    const idx = try g.astype(try g.sliceAxis(picked, 1, 0, w.top_k), .uint32);
    const top = try g.takeAlong(scores, idx, 1);
    return .{ .idx = idx, .weights = try g.mulScalar(try g.div(top, try g.addScalar(try g.sum(top, -1, true), 1e-20)), w.route_scale) };
}

/// Row-expert pairs from which the pairs are sorted by expert, so each expert's rows stream once (mlx-lm's bar).
const SORT_PAIRS = 64;

/// Each row's top-k expert outputs `[L, k, hidden]` in `x`'s dtype and their weights `[L, k]` f32 (`sum` is the MLP).
pub const Parts = struct { out: Array, weights: Array };

/// `x` `[1, L, hidden]` through its top-k experts, weighted and summed in f32, in `x`'s dtype.
pub fn forward(w: *const Weights, g: *Graph, x: Array) !Array {
    return sum(g, try parts(w, g, x, null), ops.shape(x));
}

/// The weighted sum of `p` in f32, in the outputs' dtype, as `sh` `[1, L, hidden]`.
pub fn sum(g: *Graph, p: Parts, sh: []const c_int) !Array {
    const s = try g.sum(try g.mul(try g.astype(p.out, .float32), try g.expandDims(p.weights, -1)), 1, false);
    return g.reshape(try g.astype(s, ops.dtype(p.out)), sh);
}

/// `forward` before its weighted sum; `x32` is x widened to f32, where the caller has it.
pub fn parts(w: *const Weights, g: *Graph, x: Array, x32: ?Array) !Parts {
    const sh = ops.shape(x);
    const t = sh[0] * sh[1];
    const d = sh[2];
    const k = w.top_k;
    const flat = try g.reshape(x, &.{ t, d });

    const wide = if (x32) |a| try g.reshape(a, &.{ t, d }) else try g.astype(flat, .float32);
    const routing = try route(w, g, try g.matmul(wide, w.router));
    const idx = routing.idx;
    var weights = routing.weights;

    const ids = try g.reshape(idx, &.{t * k});
    if (w.down_scale.ctx != null) weights = try g.mul(weights, try g.take(w.down_scale, idx, 0));
    // A decode row: the bank read in place, the activation inside the gate_up kernel.
    if (t == 1) if (try experts.gateUp(g, flat, &w.gate_up, ids, if (w.gate_up_scale.ctx != null) w.gate_up_scale else null, .{ w.gate_softcap, w.up_softcap })) |act| {
        if (try experts.down(g, act, &w.down, ids)) |out| return .{ .out = try g.reshape(out, &.{ t, k, d }), .weights = weights };
    };
    const sort = t * k >= SORT_PAIRS;
    var sorted = ids;
    var inv: Array = .{};
    var rows: Array = undefined;
    const experts_n = ops.shape(w.gate_up.w)[0];
    // Under four pairs an expert MLX gathers one pair at a time, reading each expert's weights again for every row
    // that picked it; padding to four takes its sorted row-block kernel, which reads each expert once. Below 1.5
    // pairs an expert the padding costs more than the reads it saves.
    const pad: c_int = if (sort and t * k < 4 * experts_n and 2 * t * k >= 3 * experts_n) 4 * experts_n - t * k else 0;
    if (sort and pad > 0) {
        // Padded pairs repeat the chunk's own experts on a zero row, so no other expert is read.
        const fill = try g.take(ids, try g.astype(try g.call(mlx.mlx_remainder, .{ try g.arange(0, @floatFromInt(pad), 1, .int32), try g.adopt(mlx.mlx_array_new_int(t * k)) }), .uint32), 0);
        const ids_p = try g.concat(&.{ ids, fill }, 0);
        const flat_p = try g.concat(&.{ flat, try g.zeros(&.{ 1, d }, ops.dtype(flat)) }, 0);
        const order = try g.argsort(ids_p, 0);
        inv = try g.argsort(order, 0);
        sorted = try g.take(ids_p, order, 0);
        // A padded pair's row is the zero row (index t).
        const pair_row = try g.minimum(try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(k)) }), try g.astype(try g.adopt(mlx.mlx_array_new_int(t)), .uint32));
        rows = try g.reshape(try g.take(flat_p, pair_row, 0), &.{ t * k + pad, 1, d });
    } else if (sort) {
        const order = try g.argsort(ids, 0);
        inv = try g.argsort(order, 0);
        sorted = try g.take(ids, order, 0);
        const src = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(k)) });
        rows = try g.reshape(try g.take(flat, src, 0), &.{ t * k, 1, d });
    } else {
        rows = try g.reshape(try g.broadcast(try g.reshape(flat, &.{ t, 1, d }), &.{ t, k, d }), &.{ t * k, 1, d });
    }

    const np = t * k + pad;
    const gu = try gather(g, rows, &w.gate_up, sorted, sort);
    const inner = @divExact(ops.shape(gu)[2], 2);
    const pairs = try g.reshape(gu, &.{ np, inner, 2 });
    const scale: ?Array = if (w.gate_up_scale.ctx != null) try g.reshape(try g.take(w.gate_up_scale, sorted, 0), &.{ np, 1, 1 }) else null;
    const act = try compiled.softcappedSwiglu(g, try g.sliceAxis(pairs, 2, 0, 1), try g.sliceAxis(pairs, 2, 1, 2), scale, w.gate_softcap, w.up_softcap);
    const down = try gather(g, try g.reshape(act, &.{ np, 1, inner }), &w.down, sorted, sort);

    const unsorted = if (sort) try g.take(try g.reshape(down, &.{ np, d }), inv, 0) else down;
    const out = try g.reshape(if (pad > 0) try g.sliceAxis(unsorted, 0, 0, t * k) else unsorted, &.{ t, k, d });
    return .{ .out = out, .weights = weights };
}

// ── tests ──

const std = @import("std");
const testing = std.testing;

// Bar: a short chunk padded to four pairs an expert gives each real pair the output MLX's one-pair-at-a-time gather
// gives it, to a bf16 ulp or two (the row-block kernel sums in another order).
test "inkling2 moe: a short chunk's padded pairs leave every real pair's output as the per-pair gather's" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const E = 128;
    const H = 256;
    const I = 128;
    for ([_][:0]const u8{ "nvfp4", "mxfp8" }, 0..) |m, n| {
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, n)});
        var w: Weights = .{
            .router = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ H, E }, @as(usize, 2), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key }),
            .bias = try g.zeros(&.{E}, .float32),
            .gate_up = try experts.quantBank(&g, E, 2 * I, H, m, 10 * n + 1),
            .down = try experts.quantBank(&g, E, H, I, m, 10 * n + 2),
            .top_k = 8,
            .route_scale = 2,
            .gate_softcap = 4,
            .up_softcap = 25,
        };
        w.gate_up_scale = try g.addScalar(try g.call(mlx.mlx_random_uniform, .{ try g.scalar(0, .float32), try g.scalar(1, .float32), &[_]c_int{E}, @as(usize, 1), ops.Dtype.float32, key }), 0.5);
        for ([_]c_int{ 24, 31, 63 }) |t| {
            const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, t, H }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key }), .bfloat16);
            const got = try parts(&w, &g, x, null);
            // The same pairs one at a time: unsorted, each its own row.
            const flat = try g.reshape(x, &.{ t, H });
            const r = try route(&w, &g, try g.matmul(try g.astype(flat, .float32), w.router));
            const ids = try g.reshape(r.idx, &.{t * 8});
            const rows = try g.reshape(try g.broadcast(try g.reshape(flat, &.{ t, 1, H }), &.{ t, 8, H }), &.{ t * 8, 1, H });
            const pairs = try g.reshape(try gather(&g, rows, &w.gate_up, ids, false), &.{ t * 8, I, 2 });
            const sc = try g.reshape(try g.take(w.gate_up_scale, ids, 0), &.{ t * 8, 1, 1 });
            const act = try compiled.softcappedSwiglu(&g, try g.sliceAxis(pairs, 2, 0, 1), try g.sliceAxis(pairs, 2, 1, 2), sc, 4, 25);
            const want = try g.reshape(try gather(&g, try g.reshape(act, &.{ t * 8, 1, I }), &w.down, ids, false), &.{ t, 8, H });
            const a = try g.astype(got.out, .float32);
            const b = try g.astype(want, .float32);
            const rel = try g.call(mlx.mlx_max, .{ try g.div(try g.abs(try g.sub(a, b)), try g.addScalar(try g.abs(b), 1e-2)), false });
            try ops.eval(rel);
            var v: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&v, rel));
            try testing.expect(v <= 2.0 / 128.0);
        }
    }
}
