//! Inkling 2's L2 clip in one kernel: the op chain is a scaled cast, a square, a sum and the clip, four dependent
//! dispatches, three times a layer visit (q, k and the relative states).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Graph = ops.Graph;
const Array = ops.Array;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/l2_clip.metal");
pub const INPUTS = [_][*:0]const u8{ "x", "scale", "threshold" };
pub const OUTPUTS = [_][*:0]const u8{"out"};

/// Threads a threadgroup: eight vectors of one simdgroup each.
pub const TG = 256;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_l2_clip", in_vec, out_vec, SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The launch for `vectors` vectors of `d` over `heads` scale rows: grid, threadgroup.
pub fn grid(vectors: usize) [2]c_int {
    return .{ @intCast(vectors * 32), @intCast(@min(TG, vectors * 32)) };
}

/// `x` `[..., H, D]` times `scale` `[H, D]` (null = 1), shrunk to L2 norm `threshold` over D, in x's dtype. Null where
/// it declines: off Metal, an empty or non-float input, a D past 1024.
pub fn fused(g: *Graph, x: Array, scale: ?Array, threshold: f32) !?Array {
    if (!mlx.streamIsGpu(g.s)) return null;
    const sh = ops.shape(x);
    if (sh.len < 2) return null;
    const dt = ops.dtype(x);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    const d = sh[sh.len - 1];
    const h = sh[sh.len - 2];
    if (d < 1 or d > 1024) return null;
    if (scale) |s| if (!std.mem.eql(c_int, ops.shape(s), &.{ h, d })) return null;
    var vectors: usize = 1;
    for (sh[0 .. sh.len - 1]) |n| vectors *= @intCast(n);
    if (vectors == 0) return null;
    const k = getKernel() catch return null;
    const launch = grid(vectors);

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, launch[0], 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, launch[1], 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "SCALED", scale != null));

    const thr = try g.fromSlice(f32, &.{threshold}, &.{1});
    // Unscaled, the scale slot binds the threshold; the kernel never reads it.
    const inputs = [_]Array{ x, scale orelse thr, thr };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, g.s));
    var out = mlx.mlx_array_new();
    const rc = mlx.mlx_vector_array_get(&out, out_vec, 0);
    const kept = try g.adopt(out);
    try mlx.check(rc);
    return kept;
}

// ── tests ──

const testing = std.testing;
const compiled = @import("compiled.zig");

// Bar: the kernel gives the compiled chain's values (to f32 sum order) at the pack's head widths, scaled or not.
test "inkling2 l2_clip: the kernel gives the op chain's values" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const cases = [_]struct { rows: c_int, h: c_int, d: c_int, scaled: bool, dt: ops.Dtype }{
        .{ .rows = 1, .h = 24, .d = 128, .scaled = true, .dt = .bfloat16 },
        .{ .rows = 1, .h = 8, .d = 128, .scaled = false, .dt = .bfloat16 },
        .{ .rows = 37, .h = 24, .d = 16, .scaled = true, .dt = .bfloat16 },
        .{ .rows = 5, .h = 3, .d = 40, .scaled = true, .dt = .float32 },
    };
    for (cases, 0..) |cs, n| {
        const x = try normal(&g, &.{ 1, cs.rows, cs.h, cs.d }, 10 * n, 2, cs.dt);
        const scale: ?Array = if (cs.scaled) try normal(&g, &.{ cs.h, cs.d }, 10 * n + 1, 1, .bfloat16) else null;
        const got = (try fused(&g, x, scale, 5)) orelse return error.KernelDeclined;
        const want = try compiled.l2Clip(&g, x, scale, 5);
        try testing.expectEqual(ops.dtype(want), ops.dtype(got));
        try testing.expectEqualSlices(c_int, ops.shape(want), ops.shape(got));
        const diff = try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(try g.astype(got, .float32), try g.astype(want, .float32))), false });
        try ops.eval(diff);
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, diff));
        // One bf16 ulp at the clipped values' size (|x| up to 5).
        try testing.expect(worst <= @as(f32, if (cs.dt == .float32) 1e-5 else 5.0 / 128.0));
    }
}

fn normal(g: *Graph, sh: []const c_int, seed: u64, scale: f32, dt: ops.Dtype) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), scale, key }), dt);
    try ops.eval(x);
    return x;
}

// Bar: at prefill widths every output is within one bf16 ulp of the chain run in f32 on the CPU, and rounds the same
// way in all but a sliver of ties.
test "inkling2 l2_clip: prefill widths match the CPU chain to an ulp" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var cpu = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer cpu.deinit();
    const cases = [_]struct { rows: c_int, h: c_int, d: c_int, scaled: bool, spread: f32 }{
        .{ .rows = 2048, .h = 24, .d = 128, .scaled = true, .spread = 0.45 },
        .{ .rows = 8192, .h = 24, .d = 128, .scaled = true, .spread = 3 },
        .{ .rows = 2048, .h = 8, .d = 128, .scaled = false, .spread = 0.45 },
        .{ .rows = 8192, .h = 24, .d = 16, .scaled = true, .spread = 1.5 },
        .{ .rows = 2048, .h = 24, .d = 16, .scaled = true, .spread = 0.2 },
    };
    for (cases, 0..) |cs, n| {
        var t = g.scope();
        defer t.deinit();
        var c = cpu.scope();
        defer c.deinit();
        // `spread` puts the vectors' norms near the threshold (0.45 x sqrt(128) ~ 5) or well past it.
        const x = try normal(&t, &.{ 1, cs.rows, cs.h, cs.d }, 100 + n, cs.spread, .bfloat16);
        const scale: ?Array = if (cs.scaled) try normal(&t, &.{ cs.h, cs.d }, 200 + n, 1, .bfloat16) else null;
        const got = (try fused(&t, x, scale, 5)) orelse return error.KernelDeclined;
        const st = try c.mul(try c.astype(x, .float32), if (scale) |sc| try c.astype(sc, .float32) else try c.scalar(1, .float32));
        const norm = try c.sqrt(try c.sum(try c.square(st), -1, true));
        const factor = try c.minimum(try c.div(try c.scalar(5, .float32), try c.addScalar(norm, 1e-6)), try c.scalar(1, .float32));
        const exact = try c.mul(st, factor);
        const want = try c.astype(exact, .bfloat16);
        const g32 = try c.astype(got, .float32);
        const w32 = try c.astype(want, .float32);
        const off = try c.abs(try c.sub(g32, exact));
        // One bf16 ulp of the exact value, with a floor for values near zero.
        const ulp = try c.addScalar(try c.mulScalar(try c.abs(exact), 1.0 / 128.0), 1e-30);
        const worst = try c.call(mlx.mlx_max, .{ try c.div(off, ulp), false });
        const differ = try c.call(mlx.mlx_sum, .{ try c.astype(try c.call(mlx.mlx_greater, .{ try c.abs(try c.sub(g32, w32)), try c.scalar(0, .float32) }), .float32), false });
        try ops.evalMany(&.{ worst, differ });
        var wv: f32 = 0;
        var dv: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&wv, worst));
        try mlx.check(mlx.mlx_array_item_float32(&dv, differ));
        const total: f32 = @floatFromInt(cs.rows * cs.h * cs.d);
        if (wv > 1.0 or dv > total * 1e-3) std.debug.print("l2_clip prefill rows={d} h={d} d={d} scaled={}: worst {d:.3} ulp, {d:.0} of {d:.0} differ from the CPU chain\n", .{ cs.rows, cs.h, cs.d, cs.scaled, wv, dv, total });
        try testing.expect(wv <= 1.0);
        try testing.expect(dv <= total * 1e-3);
    }
}
