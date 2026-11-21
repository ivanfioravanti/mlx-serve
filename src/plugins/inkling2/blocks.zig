//! Inkling 2's small blocks, as mlx-lm's `inkling2/layers.py` computes them: the unweighted RMS norm, the L2 clip,
//! depth attention over the block bank, the loop's message gate, and the static and hidden-conditioned short convs.
//! Everything here runs in f32 and casts back where the reference does.

const std = @import("std");
const ops = @import("../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub fn f32Of(g: *Graph, x: Array) !Array {
    return if (ops.dtype(x) == .float32) x else g.astype(x, .float32);
}

/// RMS norm with no gain, in `x`'s dtype.
pub fn rms(g: *Graph, x: Array, eps: f32) !Array {
    return g.rmsNorm(x, .{}, eps);
}

const compiled = @import("compiled.zig");
pub const l2Clip = compiled.l2Clip;

/// Learned queries `[Q, H]` attend over the depth sources of each token (the `bank`'s block states, then `partial`,
/// all `[1, L, H]` f32, which are also the values): `[1, L, Q, H]`.
pub fn depth(g: *Graph, bank: []const Array, partial: Array, query: Array) !Array {
    if (ops.shape(query)[0] == 1) {
        const sh = ops.shape(partial);
        return g.reshape(try depthOne(g, bank, partial, query), &.{ sh[0], sh[1], 1, sh[2] });
    }
    if (try @import("depth_attn.zig").fused(g, bank, partial, query)) |out| return out;
    return depthChain(g, bank, partial, query);
}

/// `depth` for one query `[1, H]`: `[1, L, H]`.
pub fn depthOne(g: *Graph, bank: []const Array, partial: Array, query: Array) !Array {
    if (try @import("depth_attn.zig").fused(g, bank, partial, query)) |out| return out;
    return row(g, try depthChain(g, bank, partial, query), 0);
}

/// `depth` as ops: the sources stacked, scored, softmaxed and mixed.
pub fn depthChain(g: *Graph, bank: []const Array, partial: Array, query: Array) !Array {
    var parts: [64]Array = undefined;
    std.debug.assert(bank.len < parts.len);
    for (bank, 0..) |x, i| parts[i] = try g.expandDims(x, 2);
    parts[bank.len] = try g.expandDims(partial, 2);
    const stacked = try g.concat(parts[0 .. bank.len + 1], 2);
    const h: f32 = @floatFromInt(ops.shape(partial)[2]);
    const scores = try g.mulScalar(try g.matmul(stacked, try g.transpose(query, &.{ 1, 0 })), 1.0 / @sqrt(h));
    const w = try g.softmax(scores, 2, true);
    return g.matmul(try g.transpose(w, &.{ 0, 1, 3, 2 }), stacked);
}

/// Row `i` of `x` `[1, L, Q, H]` as `[1, L, H]`.
pub fn row(g: *Graph, x: Array, i: c_int) !Array {
    const sh = ops.shape(x);
    return g.reshape(try g.sliceAxis(x, 2, i, i + 1), &.{ sh[0], sh[1], sh[3] });
}

/// A loop message mixed into a layer input. Weights are stored `[in, out]`.
pub const MessageGate = struct {
    /// `[2H, rank]` in the checkpoint's dtype.
    down: Array,
    /// `[rank, H]` f32.
    forget: Array,
    message: Array,

    /// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, both gates read off the RMS-normed pair; f32.
    pub fn forward(w: *const MessageGate, g: *Graph, hidden: Array, message: Array) !Array {
        const h = try f32Of(g, hidden);
        const pair = try g.concat(&.{ try rms(g, h, 1e-6), try rms(g, message, 1e-6) }, 2);
        const st = try f32Of(g, try g.matmul(try g.astype(pair, ops.dtype(w.down)), w.down));
        return compiled.gateMix(g, try g.matmul(st, w.forget), try g.matmul(st, w.message), h, message);
    }
};

/// The conv input: the `k - 1` rows before the chunk (`prev`, zeros at the sequence start) then `x`, f32.
fn window(g: *Graph, x32: Array, prev: ?Array, k: c_int) !Array {
    const sh = ops.shape(x32);
    const before = prev orelse try g.zeros(&.{ sh[0], k - 1, sh[2] }, .float32);
    return g.concat(&.{ before, x32 }, 1);
}

/// The last `k - 1` rows of a conv input, copied so the state does not hold the chunk.
fn tail(g: *Graph, win: Array, k: c_int) !Array {
    const n = ops.shape(win)[1];
    return g.contiguous(try g.sliceAxis(win, 1, n - (k - 1), n));
}

/// `input` is the whole conv input `[1, K-1+L, C]` (what a verify forward leaves for the rollback).
pub const Conv = struct { out: Array, state: Array, input: Array };

/// Depthwise causal conv `w` `[C, K, 1]` in the weight's dtype plus the input, over `sqrt(K + 1)`.
pub fn sconv(g: *Graph, x: Array, w: Array, prev: ?Array) !Conv {
    const k = ops.shape(w)[1];
    const x32 = try f32Of(g, x);
    const win = try window(g, x32, prev, k);
    const y = try g.conv1d(try g.astype(win, ops.dtype(w)), w, 1, 0, 1, ops.shape(x)[2]);
    const out = try compiled.residualScale(g, y, x, 1.0 / @sqrt(@as(f32, @floatFromInt(k + 1))));
    return .{ .out = out, .state = try tail(g, win, k), .input = win };
}

/// A global layer's conv on k or v: per-head taps `static` `[K, C]` plus `dynamic` `[1, L, heads * K]` (the layer
/// input's projection), plus the input, over `sqrt(2K + 1)`.
pub fn dynamicSconv(g: *Graph, x: Array, dynamic: Array, static: Array, heads: c_int, prev: ?Array) !Conv {
    const sh = ops.shape(x);
    const l = sh[1];
    const c = sh[2];
    const k = ops.shape(static)[0];
    const d = @divExact(c, heads);
    const x32 = try f32Of(g, x);
    const dyn = try g.reshape(try f32Of(g, dynamic), &.{ sh[0], l, heads, 1, k });
    const st = try g.reshape(try g.transpose(try f32Of(g, static), &.{ 1, 0 }), &.{ heads, d, k });
    const taps = try g.add(st, dyn);
    const win = try window(g, x32, prev, k);
    var lags: [64]Array = undefined;
    for (0..@intCast(k)) |lag| {
        const from = k - 1 - @as(c_int, @intCast(lag));
        lags[lag] = try g.expandDims(try g.sliceAxis(win, 1, from, from + l), 3);
    }
    const wins = try g.reshape(try g.concat(lags[0..@intCast(k)], 3), &.{ sh[0], l, heads, d, k });
    const y = try g.reshape(try g.sum(try g.mul(wins, taps), -1, false), &.{ sh[0], l, c });
    const out = try compiled.residualScale(g, y, x, 1.0 / @sqrt(@as(f32, @floatFromInt(2 * k + 1))));
    return .{ .out = out, .state = try tail(g, win, k), .input = win };
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../mlx/ffi.zig");

fn host(x: Array) ![]const f32 {
    try ops.eval(x);
    var n: usize = 1;
    for (ops.shape(x)) |dim| n *= @intCast(dim);
    return (mlx.mlx_array_data_float32(x) orelse return error.NoData)[0..n];
}

test "inkling2 blocks: a short conv fed row by row with its state equals the whole chunk" {
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var xs: [6 * 4]f32 = undefined;
    var ws: [4 * 4]f32 = undefined;
    var ds: [6 * 2 * 4]f32 = undefined;
    for (&xs) |*v| v.* = prng.random().floatNorm(f32);
    for (&ws) |*v| v.* = prng.random().floatNorm(f32);
    for (&ds) |*v| v.* = prng.random().floatNorm(f32);
    const x = try g.fromSlice(f32, &xs, &.{ 1, 6, 4 });
    const w = try g.fromSlice(f32, &ws, &.{ 4, 4, 1 });
    const static = try g.fromSlice(f32, &ws, &.{ 4, 4 });
    const dyn = try g.fromSlice(f32, &ds, &.{ 1, 6, 8 });
    const whole = try sconv(&g, x, w, null);
    const whole_dyn = try dynamicSconv(&g, x, dyn, static, 2, null);
    var st: ?Array = null;
    var st_dyn: ?Array = null;
    for (0..6) |t| {
        const i: c_int = @intCast(t);
        const step = try sconv(&g, try g.sliceAxis(x, 1, i, i + 1), w, st);
        const step_dyn = try dynamicSconv(&g, try g.sliceAxis(x, 1, i, i + 1), try g.sliceAxis(dyn, 1, i, i + 1), static, 2, st_dyn);
        st = step.state;
        st_dyn = step_dyn.state;
        for (try host(step.out), (try host(whole.out))[t * 4 ..][0..4]) |a, b| try testing.expectApproxEqAbs(b, a, 1e-5);
        for (try host(step_dyn.out), (try host(whole_dyn.out))[t * 4 ..][0..4]) |a, b| try testing.expectApproxEqAbs(b, a, 1e-5);
    }
    try testing.expectEqualSlices(f32, try host(whole.state), try host(st.?));
}

test "inkling2 blocks: depth attention of a zero query averages the sources; the clip caps the norm" {
    var g = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const a = try g.fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 2 });
    const b = try g.fromSlice(f32, &.{ 3, 6, 5, 0 }, &.{ 1, 2, 2 });
    const out = try depth(&g, &.{a}, b, try g.zeros(&.{ 1, 2 }, .float32));
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 1, 2 }, ops.shape(out));
    try testing.expectEqualSlices(f32, &.{ 2, 4, 4, 2 }, try host(out));
    const clipped = try host(try l2Clip(&g, try g.fromSlice(f32, &.{ 3, 4, 0.3, 0.4 }, &.{ 2, 2 }), null, 1));
    try testing.expectApproxEqAbs(@as(f32, 0.6), clipped[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.8), clipped[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.3), clipped[2], 1e-6);
}
