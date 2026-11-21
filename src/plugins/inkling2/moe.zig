//! Inkling 2's routed MLP (mlx-lm `SparseMoeBlock`): an f32 sigmoid router with a selection-only bias, top-k
//! weights normalized and scaled, experts whose fused `gate_up_proj` rows alternate gate and up, a softcapped SwiGLU,
//! no shared expert. The bank's quant is its own (NVFP4 with a per-expert global scale, MXFP8, affine or dense).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const compiled = @import("compiled.zig");
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
    const sort = t * k >= SORT_PAIRS;
    var sorted = ids;
    var inv: Array = .{};
    var rows: Array = undefined;
    if (sort) {
        const order = try g.argsort(ids, 0);
        inv = try g.argsort(order, 0);
        sorted = try g.take(ids, order, 0);
        const src = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(k)) });
        rows = try g.reshape(try g.take(flat, src, 0), &.{ t * k, 1, d });
    } else {
        rows = try g.reshape(try g.broadcast(try g.reshape(flat, &.{ t, 1, d }), &.{ t, k, d }), &.{ t * k, 1, d });
    }

    const gu = try gather(g, rows, &w.gate_up, sorted, sort);
    const inner = @divExact(ops.shape(gu)[2], 2);
    const pairs = try g.reshape(gu, &.{ t * k, inner, 2 });
    const scale: ?Array = if (w.gate_up_scale.ctx != null) try g.reshape(try g.take(w.gate_up_scale, sorted, 0), &.{ t * k, 1, 1 }) else null;
    const act = try compiled.softcappedSwiglu(g, try g.sliceAxis(pairs, 2, 0, 1), try g.sliceAxis(pairs, 2, 1, 2), scale, w.gate_softcap, w.up_softcap);
    const down = try gather(g, try g.reshape(act, &.{ t * k, 1, inner }), &w.down, sorted, sort);

    const out = try g.reshape(if (sort) try g.take(try g.reshape(down, &.{ t * k, d }), inv, 0) else down, &.{ t, k, d });
    if (w.down_scale.ctx != null) weights = try g.mul(weights, try g.take(w.down_scale, idx, 0));
    return .{ .out = out, .weights = weights };
}
