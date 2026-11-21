//! Inkling 2's routed MLP (mlx-lm `SparseMoeBlock`): an f32 sigmoid router with a selection-only bias, top-k
//! weights normalized and scaled, experts whose fused `gate_up_proj` rows alternate gate and up, a softcapped SwiGLU,
//! no shared expert. The bank's quant is its own (NVFP4 with a per-expert global scale, MXFP8, affine or dense).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;

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

fn softcap(g: *Graph, x: Array, cap: f32) !Array {
    return g.mulScalar(try g.tanh(try g.mulScalar(x, 1.0 / cap)), cap);
}

fn gather(g: *Graph, x: Array, bank: *const Bank, rhs: Array) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, .{ .ctx = null }, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, true, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

/// `x` `[1, L, hidden]` through its top-k experts, weighted and summed in f32, in `x`'s dtype.
pub fn forward(w: *const Weights, g: *Graph, x: Array) !Array {
    const sh = ops.shape(x);
    const t = sh[0] * sh[1];
    const d = sh[2];
    const k = w.top_k;
    const dt = ops.dtype(x);
    const flat = try g.reshape(x, &.{ t, d });

    const scores = try g.sigmoid(try g.matmul(try g.astype(flat, .float32), w.router));
    const picked = try g.argpartition(try g.neg(try g.add(scores, w.bias)), k - 1, -1);
    const idx = try g.astype(try g.sliceAxis(picked, 1, 0, k), .uint32);
    const top = try g.takeAlong(scores, idx, 1);
    var weights = try g.mulScalar(try g.div(top, try g.addScalar(try g.sum(top, -1, true), 1e-20)), w.route_scale);

    // Sorted by expert, so each expert's rows stream once.
    const ids = try g.reshape(idx, &.{t * k});
    const order = try g.argsort(ids, 0);
    const inv = try g.argsort(order, 0);
    const sorted = try g.take(ids, order, 0);
    const src = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(k)) });
    const rows = try g.reshape(try g.take(flat, src, 0), &.{ t * k, 1, d });

    var gu = try g.astype(try gather(g, rows, &w.gate_up, sorted), .float32);
    if (w.gate_up_scale.ctx != null) gu = try g.mul(gu, try g.reshape(try g.take(w.gate_up_scale, sorted, 0), &.{ t * k, 1, 1 }));
    const pairs = try g.reshape(gu, &.{ t * k, @divExact(ops.shape(gu)[2], 2), 2 });
    const gate = try g.sliceAxis(pairs, 2, 0, 1);
    const up = try g.sliceAxis(pairs, 2, 1, 2);
    const act = try g.mul(try g.mul(try softcap(g, gate, w.gate_softcap), try g.sigmoid(gate)), try softcap(g, up, w.up_softcap));
    const inner = ops.shape(act)[1];
    const down = try gather(g, try g.astype(try g.reshape(act, &.{ t * k, 1, inner }), dt), &w.down, sorted);

    const out = try g.reshape(try g.take(try g.reshape(down, &.{ t * k, d }), inv, 0), &.{ t, k, d });
    if (w.down_scale.ctx != null) weights = try g.mul(weights, try g.take(w.down_scale, idx, 0));
    const sum = try g.sum(try g.mul(try g.astype(out, .float32), try g.expandDims(weights, -1)), 1, false);
    return g.reshape(try g.astype(sum, dt), sh);
}
