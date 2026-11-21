//! Inkling 2's elementwise chains as shapeless `mlx_compile` closures, as mlx-lm compiles them (`_l2_clip`,
//! `_softcapped_swiglu`): one fused kernel where the op chain is ten dispatches, 48 visits a token. A closure is
//! traced on the first GPU stream that calls it; a CPU stream, or a build with no GPU backend, runs the chain.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;

/// `body` over `n` arrays: `call` runs it through its compiled closure, or as plain ops where there is none.
fn Compiled(comptime n: usize, comptime body: fn (*Graph, [n]Array) anyerror!Array) type {
    return struct {
        var closure: mlx.mlx_closure = .{ .ctx = null };
        var stream: ops.Stream = undefined;
        var tried = false;

        fn trace(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, _: ?*anyopaque) callconv(.c) c_int {
            var g = Graph.init(std.heap.c_allocator, stream);
            defer g.deinit();
            var args: [n]Array = undefined;
            for (&args, 0..) |*a, i| {
                var x = mlx.mlx_array_new();
                const rc = mlx.mlx_vector_array_get(&x, in, i);
                a.* = g.adopt(x) catch return -1;
                if (rc != 0) return -1;
            }
            const out = body(&g, args) catch return -1;
            res.* = mlx.mlx_vector_array_new_data(&[_]Array{out}, 1);
            return 0;
        }

        pub fn call(g: *Graph, args: [n]Array) !Array {
            if (!mlx.streamOnGpuDevice(g.s)) return body(g, args);
            if (!tried) {
                tried = true;
                stream = g.s;
                const raw = mlx.mlx_closure_new_func_payload(&trace, null, null);
                defer _ = mlx.mlx_closure_free(raw);
                if (mlx.mlx_compile(&closure, raw, true) != 0) closure = .{ .ctx = null };
            }
            if (closure.ctx == null) return body(g, args);
            const in = mlx.mlx_vector_array_new_data(&args, n);
            defer _ = mlx.mlx_vector_array_free(in);
            var out = mlx.mlx_vector_array{ .ctx = null };
            defer _ = mlx.mlx_vector_array_free(out);
            try mlx.check(mlx.mlx_closure_apply(&out, closure, in));
            var r = mlx.mlx_array_new();
            const rc = mlx.mlx_vector_array_get(&r, out, 0);
            const kept = try g.adopt(r);
            try mlx.check(rc);
            return kept;
        }
    };
}

fn l2ClipBody(g: *Graph, a: [3]Array) !Array {
    const x, const scale, const threshold = a;
    const st = try g.mul(try g.astype(x, .float32), scale);
    const norm = try g.sqrt(try g.sum(try g.square(st), -1, true));
    const factor = try g.minimum(try g.div(threshold, try g.addScalar(norm, 1e-6)), try g.scalar(1, .float32));
    return g.astype(try g.mul(st, factor), ops.dtype(x));
}

/// `x * scale` (f32; null = 1) shrunk to L2 norm `threshold` over the last axis, in f32, cast back to `x`'s dtype.
pub fn l2Clip(g: *Graph, x: Array, scale: ?Array, threshold: f32) !Array {
    return Compiled(3, l2ClipBody).call(g, .{ x, scale orelse try g.scalar(1, .float32), try g.scalar(threshold, .float32) });
}

fn swigluBody(g: *Graph, a: [5]Array) !Array {
    const gate_in, const up_in, const scale, const gate_cap, const up_cap = a;
    const gate = try g.mul(try g.astype(gate_in, .float32), scale);
    const up = try g.mul(try g.astype(up_in, .float32), scale);
    const gate_capped = try g.mul(try g.tanh(try g.div(gate, gate_cap)), gate_cap);
    const up_capped = try g.mul(try g.tanh(try g.div(up, up_cap)), up_cap);
    return g.astype(try g.mul(try g.mul(gate_capped, try g.sigmoid(gate)), up_capped), ops.dtype(gate_in));
}

/// `cap_g tanh(g / cap_g) * sigmoid(g) * cap_u tanh(u / cap_u)` with `g = gate * scale`, `u = up * scale` (scale
/// f32, broadcast; null = 1), in f32, cast back to `gate`'s dtype.
pub fn softcappedSwiglu(g: *Graph, gate: Array, up: Array, scale: ?Array, gate_cap: f32, up_cap: f32) !Array {
    return Compiled(5, swigluBody).call(g, .{ gate, up, scale orelse try g.scalar(1, .float32), try g.scalar(gate_cap, .float32), try g.scalar(up_cap, .float32) });
}

fn gateMixBody(g: *Graph, a: [4]Array) !Array {
    const forget, const gain, const hidden, const message = a;
    return g.add(try g.mul(try g.sub(try g.scalar(1, .float32), try g.tanh(forget)), hidden), try g.mul(try g.tanh(gain), message));
}

/// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, all f32.
pub fn gateMix(g: *Graph, forget: Array, gain: Array, hidden: Array, message: Array) !Array {
    return Compiled(4, gateMixBody).call(g, .{ forget, gain, hidden, message });
}

fn sigmoidGateBody(g: *Graph, a: [2]Array) !Array {
    const x, const gate = a;
    return g.astype(try g.mul(try g.astype(x, .float32), try g.sigmoid(try g.astype(gate, .float32))), ops.dtype(gate));
}

/// `x * sigmoid(gate)` in f32, in `gate`'s dtype.
pub fn sigmoidGate(g: *Graph, x: Array, gate: Array) !Array {
    return Compiled(2, sigmoidGateBody).call(g, .{ x, gate });
}

fn residualScaleBody(g: *Graph, a: [3]Array) !Array {
    const y, const x, const by = a;
    return g.astype(try g.mul(try g.add(try g.astype(y, .float32), try g.astype(x, .float32)), by), ops.dtype(x));
}

/// `(y + x) * by` in f32, in `x`'s dtype.
pub fn residualScale(g: *Graph, y: Array, x: Array, by: f32) !Array {
    return Compiled(3, residualScaleBody).call(g, .{ y, x, try g.scalar(by, .float32) });
}

// ── tests ──

const testing = std.testing;

fn randn(g: *Graph, sh: []const c_int, seed: u64, dt: ops.Dtype) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    return g.astype(try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 3), key }), dt);
}

fn maxDiff(g: *Graph, a: Array, b: Array) !f32 {
    try testing.expectEqual(ops.dtype(b), ops.dtype(a));
    try testing.expectEqualSlices(c_int, ops.shape(b), ops.shape(a));
    const d = try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(try g.astype(a, .float32), try g.astype(b, .float32))), false });
    try ops.eval(d);
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, d));
    return v;
}

test "inkling2 compiled: the closures give the op chains' values, in the input's dtype" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    for ([_]ops.Dtype{ .bfloat16, .float32 }) |dt| {
        const x = try randn(&g, &.{ 1, 5, 6, 16 }, 1, dt);
        const scale = try randn(&g, &.{ 6, 16 }, 2, .float32);
        for ([_]?Array{ scale, null }) |sc| {
            const want = try l2ClipBody(&g, .{ x, sc orelse try g.scalar(1, .float32), try g.scalar(5, .float32) });
            try testing.expect(try maxDiff(&g, try l2Clip(&g, x, sc, 5), want) <= 1e-5);
        }
        const gate = try randn(&g, &.{ 8, 24, 1 }, 3, dt);
        const up = try randn(&g, &.{ 8, 24, 1 }, 4, dt);
        const per_row = try g.abs(try randn(&g, &.{ 8, 1, 1 }, 5, .float32));
        for ([_]?Array{ per_row, null }) |sc| {
            const want = try swigluBody(&g, .{ gate, up, sc orelse try g.scalar(1, .float32), try g.scalar(4, .float32), try g.scalar(25, .float32) });
            const got = try softcappedSwiglu(&g, gate, up, sc, 4, 25);
            try testing.expect(try maxDiff(&g, got, want) <= 1e-5 * 100);
        }
        const h = try randn(&g, &.{ 1, 5, 16 }, 6, .float32);
        const m = try randn(&g, &.{ 1, 5, 16 }, 7, .float32);
        try testing.expect(try maxDiff(&g, try gateMix(&g, m, h, h, m), try gateMixBody(&g, .{ m, h, h, m })) <= 1e-5);
        try testing.expect(try maxDiff(&g, try sigmoidGate(&g, gate, up), try sigmoidGateBody(&g, .{ gate, up })) <= 1e-5);
        try testing.expect(try maxDiff(&g, try residualScale(&g, gate, up, 0.5), try residualScaleBody(&g, .{ gate, up, try g.scalar(0.5, .float32) })) <= 1e-5);
    }
}
