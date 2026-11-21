//! Inkling 2's depth attention in one kernel (mlx-lm's `depth_attention`): the op chain stacks the bank, scores,
//! softmaxes and mixes in five dispatches, two or three times a layer visit.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Graph = ops.Graph;
const Array = ops.Array;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/depth_attn.metal");
pub const INPUTS = [_][*:0]const u8{ "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "partial", "queries", "scale" };
pub const OUTPUTS = [_][*:0]const u8{"out"};

/// Block states the kernel binds as their own inputs, and queries one launch answers.
pub const MAX_BANK = 8;
pub const MAX_QUERIES = 4;

/// Threads for a row of `d` values: at least eight values a thread, whole simdgroups, 256 at most.
pub fn threads(d: usize) usize {
    return @min(256, @max(32, d / 8 / 32 * 32));
}

/// Threadgroup bytes: one f32 a simdgroup for each of the `(bank + 1) * queries` scores.
pub fn tgBytes(bank: usize, queries: usize, d: usize) usize {
    return 4 * (bank + 1) * queries * (threads(d) / 32);
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_depth_attn", in_vec, out_vec, SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// `bank` and `partial` `[1, L, H]` f32, `query` `[Q, H]` f32: `[1, L, Q, H]`. Null where it declines: off Metal,
/// more block states or queries than it binds, an input that is not f32.
pub fn fused(g: *Graph, bank: []const Array, partial: Array, query: Array) !?Array {
    if (!mlx.streamIsGpu(g.s) or bank.len > MAX_BANK) return null;
    const sh = ops.shape(partial);
    const qs = ops.shape(query);
    if (sh.len != 3 or qs.len != 2 or qs[1] != sh[2] or qs[0] < 1 or qs[0] > MAX_QUERIES or sh[2] < 32) return null;
    if (ops.dtype(partial) != .float32 or ops.dtype(query) != .float32) return null;
    for (bank) |x| if (ops.dtype(x) != .float32 or !std.mem.eql(c_int, ops.shape(x), sh)) return null;
    const k = getKernel() catch return null;
    const rows = sh[0] * sh[1];
    const d = sh[2];
    const tg: c_int = @intCast(threads(@intCast(d)));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], qs[0], d }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(bank.len)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Q", qs[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", tg));

    // A slot past the bank is never read: it binds the partial state.
    var inputs: [INPUTS.len]Array = @splat(partial);
    @memcpy(inputs[0..bank.len], bank);
    inputs[MAX_BANK + 1] = query;
    inputs[MAX_BANK + 2] = try g.fromSlice(f32, &.{1.0 / @sqrt(@as(f32, @floatFromInt(d)))}, &.{1});
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
const blocks = @import("blocks.zig");

test "inkling2 depth_attn: the kernel gives the op chain's rows at every bank size and query count" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const cases = [_]struct { bank: usize, rows: c_int, d: c_int, q: c_int }{
        .{ .bank = 0, .rows = 1, .d = 2304, .q = 1 },
        .{ .bank = 4, .rows = 1, .d = 2304, .q = 3 },
        .{ .bank = 8, .rows = 37, .d = 2304, .q = 4 },
        .{ .bank = 3, .rows = 5, .d = 100, .q = 2 },
        .{ .bank = 1, .rows = 2, .d = 32, .q = 1 },
    };
    for (cases, 0..) |cs, n| {
        var bank: [MAX_BANK]Array = undefined;
        const sh = [_]c_int{ 1, cs.rows, cs.d };
        for (bank[0..cs.bank], 0..) |*x, i| x.* = try normal(&g, &sh, 100 * n + i, 1);
        const partial = try normal(&g, &sh, 100 * n + 50, 1);
        // Queries far from zero, so the weights are far from uniform.
        const query = try normal(&g, &.{ cs.q, cs.d }, 100 * n + 51, 4);
        const got = (try fused(&g, bank[0..cs.bank], partial, query)) orelse return error.KernelDeclined;
        const want = try blocks.depthChain(&g, bank[0..cs.bank], partial, query);
        try testing.expectEqualSlices(c_int, ops.shape(want), ops.shape(got));
        const diff = try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(got, want)), false });
        try ops.eval(diff);
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, diff));
        try testing.expect(worst <= 2e-5);
    }
    var nine: [MAX_BANK + 1]Array = @splat(try normal(&g, &.{ 1, 2, 64 }, 1, 1));
    try testing.expectEqual(@as(?Array, null), try fused(&g, &nine, nine[0], try normal(&g, &.{ 1, 64 }, 2, 1)));
    try testing.expectEqual(@as(?Array, null), try fused(&g, nine[0..1], nine[0], try normal(&g, &.{ 5, 64 }, 2, 1)));
    try testing.expectEqual(@as(?Array, null), try fused(&g, nine[0..1], try g.astype(nine[0], .bfloat16), try normal(&g, &.{ 1, 64 }, 2, 1)));
}

fn normal(g: *Graph, sh: []const c_int, seed: u64, scale: f32) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), scale, key });
    try ops.eval(x);
    return x;
}
