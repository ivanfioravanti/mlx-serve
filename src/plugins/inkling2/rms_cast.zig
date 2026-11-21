//! Inkling 2's RMS norms that feed a projection, in one dispatch: the op chain is a cast in, the norm and a cast
//! out, on a layer visit's serial path (the q/k/v inputs, v's own norm, the MLP's input and its router's f32 copy).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Graph = ops.Graph;
const Array = ops.Array;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/rms_cast.metal");
pub const INPUTS = [_][*:0]const u8{ "x", "w", "eps" };
pub const OUTPUTS = [_][*:0]const u8{ "out", "twin" };
/// The widest row it takes (four values a thread, one threadgroup a row).
pub const MAX_D = 4096;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_rms_cast", in_vec, out_vec, SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// MLX's single-row launch: a thread per four values, whole simdgroups.
pub fn threads(d: usize) usize {
    return (d / 4 + @intFromBool(d % 4 != 0) + 31) / 32 * 32;
}

pub const Out = struct { out: Array, twin: ?Array };

/// `x` `[..., D]` RMS-normed over D in f32, times `w` `[D]` f32 (null = none), cast to `to`; `twin` also returns the
/// cast output widened to f32. Null where it declines: off Metal, D past `MAX_D`, a gain that is not f32.
pub fn fused(g: *Graph, x: Array, w: ?Array, eps: f32, to: ops.Dtype, twin: bool) !?Out {
    if (!mlx.streamIsGpu(g.s)) return null;
    const sh = ops.shape(x);
    if (sh.len < 1) return null;
    const d = sh[sh.len - 1];
    if (d < 1 or d > MAX_D) return null;
    if (w) |a| if (ops.dtype(a) != .float32 or !std.mem.eql(c_int, ops.shape(a), &.{d})) return null;
    const dt = ops.dtype(x);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |n| rows *= n;
    if (rows == 0) return null;
    const k = getKernel() catch return null;
    const tg: c_int = @intCast(threads(@intCast(d)));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, to));
    try mlx.check(if (twin) mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, .float32) else mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", to));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "W", w != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "TWIN", twin));

    const e = try g.fromSlice(f32, &.{eps}, &.{1});
    // Without a gain its slot binds eps; the kernel never reads it.
    const inputs = [_]Array{ x, w orelse e, e };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, g.s));
    var res: [2]Array = undefined;
    for (&res, 0..) |*r, i| {
        var a = mlx.mlx_array_new();
        const rc = mlx.mlx_vector_array_get(&a, out_vec, i);
        r.* = try g.adopt(a);
        try mlx.check(rc);
    }
    return .{ .out = res[0], .twin = if (twin) res[1] else null };
}

// ── tests ──

const testing = std.testing;

// Bar: MLX's norm and casts on the same input: the rounded output to one ulp (a sum ordered differently), the twin
// exactly the output widened.
test "inkling2 rms_cast: the kernel gives MLX's norm and casts" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const cases = [_]struct { rows: c_int, d: c_int, dt: ops.Dtype, gain: bool }{
        .{ .rows = 1, .d = 2304, .dt = .float32, .gain = true },
        .{ .rows = 8, .d = 128, .dt = .bfloat16, .gain = false },
        .{ .rows = 5, .d = 2302, .dt = .float32, .gain = true },
        .{ .rows = 3, .d = 37, .dt = .float32, .gain = false },
    };
    for (cases, 0..) |cs, n| {
        const x = try normal(&g, &.{ 1, cs.rows, cs.d }, 3 * n, 2, cs.dt);
        const w: ?Array = if (cs.gain) try normal(&g, &.{cs.d}, 3 * n + 1, 1, .float32) else null;
        const got = (try fused(&g, x, w, 1e-6, .bfloat16, true)) orelse return error.KernelDeclined;
        const want = try g.astype(try g.rmsNorm(try g.astype(x, .float32), w orelse .{}, 1e-6), .bfloat16);
        const off = try g.abs(try g.sub(try g.astype(got.out, .float32), try g.astype(want, .float32)));
        const rel = try g.call(mlx.mlx_max, .{ try g.div(off, try g.addScalar(try g.abs(try g.astype(want, .float32)), 1e-30)), false });
        const twin = try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(got.twin.?, try g.astype(got.out, .float32))), false });
        try ops.evalMany(&.{ rel, twin });
        var r: f32 = 0;
        var tw: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&r, rel));
        try mlx.check(mlx.mlx_array_item_float32(&tw, twin));
        try testing.expect(r <= 1.0 / 128.0);
        try testing.expectEqual(@as(f32, 0), tw);
    }
}

fn normal(g: *Graph, sh: []const c_int, seed: u64, scale: f32, dt: ops.Dtype) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), scale, key }), dt);
    try ops.eval(x);
    return x;
}
