//! Inkling 2's loop message gate in two kernels for a decode row or a short verify window: the op chain is nine
//! dispatches a gate, four gates a looped visit. Both launch one group an output, so no thread's own work bounds
//! the dispatch.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Graph = ops.Graph;
const Array = ops.Array;

pub const PROJ_SOURCE = kernelSource("plugins/inkling2/kernels/message_gate.metal");
pub const PROJ_HEADER = kernelSource("plugins/inkling2/kernels/block_sum.metal");
pub const PROJ_INPUTS = [_][*:0]const u8{ "hidden", "messages", "sel", "down" };
pub const MIX_SOURCE = kernelSource("plugins/inkling2/kernels/message_gate_mix.metal");
pub const MIX_INPUTS = [_][*:0]const u8{ "st", "forget", "gain", "hidden", "messages", "sel" };

/// Threads summing one projection output.
pub const TG = 256;
/// Each kernel reads every weight once a row: past a few rows the op chain's GEMMs win.
pub const MAX_ROWS = 16;

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

fn getKernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[which]) |k| return k;
    const inputs: []const [*:0]const u8 = if (which == 0) &PROJ_INPUTS else &MIX_INPUTS;
    const outputs = [_][*:0]const u8{if (which == 0) "st" else "out"};
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = if (which == 0)
        mlx.mlx_fast_metal_kernel_new("inkling2_message_gate", in_vec, out_vec, PROJ_SOURCE, PROJ_HEADER, true, false)
    else
        mlx.mlx_fast_metal_kernel_new("inkling2_message_gate_mix", in_vec, out_vec, MIX_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[which] = k;
    return k;
}

fn float(dt: ops.Dtype) bool {
    return dt == .float32 or dt == .bfloat16 or dt == .float16;
}

fn launch(g: *Graph, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Array) !Array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
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

/// `hidden` `[1, L, H]`, `messages` `[1, L, Q, H]` f32 whose row `j` is the gate's message, `down` `[rank, 2H]`,
/// `forget` and `gain` `[H, rank]` f32: `[1, L, H]` f32. Null where it declines: off Metal, more than `MAX_ROWS`
/// rows, a weight or input dtype it does not read.
pub fn fused(g: *Graph, hidden: Array, messages: Array, j: c_int, down: Array, forget: Array, gain: Array) !?Array {
    if (!mlx.streamIsGpu(g.s)) return null;
    const sh = ops.shape(hidden);
    const ms = ops.shape(messages);
    const ds = ops.shape(down);
    const fs = ops.shape(forget);
    if (sh.len != 3 or ms.len != 4 or ds.len != 2) return null;
    const rows = sh[0] * sh[1];
    const h = sh[2];
    const rank = ds[0];
    if (rows < 1 or rows > MAX_ROWS or h < TG or rank < 1) return null;
    if (ms[0] != sh[0] or ms[1] != sh[1] or ms[3] != h or j < 0 or j >= ms[2] or ds[1] != 2 * h) return null;
    if (!std.mem.eql(c_int, fs, &.{ h, rank }) or !std.mem.eql(c_int, ops.shape(gain), &.{ h, rank })) return null;
    if (!float(ops.dtype(hidden)) or !float(ops.dtype(down))) return null;
    if (ops.dtype(messages) != .float32 or ops.dtype(forget) != .float32 or ops.dtype(gain) != .float32) return null;
    const proj = getKernel(0) catch return null;
    const mix = getKernel(1) catch return null;
    const sel = try g.fromSlice(i32, &.{j}, &.{1});

    const pc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(pc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(pc, &[_]c_int{ rows, rank }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(pc, rows * rank * TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(pc, TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(pc, "TD", ops.dtype(down)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "H", h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "R", rank));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "NQ", ms[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(pc, "TG", TG));
    const st = try launch(g, proj, pc, &.{ hidden, messages, sel, down });

    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, sh.ptr, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, rows * h * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "H", h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "R", rank));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "NQ", ms[2]));
    return try launch(g, mix, mc, &.{ st, forget, gain, hidden, messages, sel });
}

// ── tests ──

const testing = std.testing;
const blocks = @import("blocks.zig");

fn normal(g: *Graph, sh: []const c_int, seed: u64, scale: f32, dt: ops.Dtype) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), scale, key }), dt);
    try ops.eval(x);
    return x;
}

/// `[in, out]` as the checkpoint's `[out, in]`.
fn outIn(g: *Graph, x: Array) !Array {
    const y = try g.contiguous(try g.transpose(x, &.{ 1, 0 }));
    try ops.eval(y);
    return y;
}

/// The op chain in f32 on the CPU, its two roundings to `down`'s dtype written out: a CPU bf16 matmul and a
/// tensor-unit GPU's f32 one are both less exact than this.
fn chain(c: *Graph, gate: *const blocks.MessageGate, hidden: Array, message: Array) !Array {
    const dt = ops.dtype(gate.down);
    const h = try c.astype(hidden, .float32);
    const pair = try c.concat(&.{ try blocks.rms(c, h, 1e-6), try blocks.rms(c, message, 1e-6) }, 2);
    const rounded = try c.astype(try c.astype(pair, dt), .float32);
    const st = try c.astype(try c.astype(try c.matmul(rounded, try c.transpose(try c.astype(gate.down, .float32), &.{ 1, 0 })), dt), .float32);
    const one = try c.scalar(1, .float32);
    return c.add(try c.mul(try c.sub(one, try c.tanh(try c.matmul(st, try c.transpose(gate.forget, &.{ 1, 0 })))), h), try c.mul(try c.tanh(try c.matmul(st, try c.transpose(gate.message, &.{ 1, 0 }))), message));
}

// Bar: a rounding of the projection that lands the other way moves a few outputs by its ulp, so the mean is held
// tight and the worst element loosely.
test "inkling2 message_gate: the kernels give the op chain's rows for each message of a short window" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var cpu = Graph.init(testing.allocator, mlx.mlx_default_cpu_stream_new());
    defer cpu.deinit();
    const h = 2304;
    for ([_]c_int{ 1, 5 }, 0..) |rows, n| for ([_]ops.Dtype{ .bfloat16, .float32 }) |dt| {
        const hidden = try normal(&g, &.{ 1, rows, h }, 10 * n + 1, 2, dt);
        const messages = try normal(&g, &.{ 1, rows, 4, h }, 10 * n + 2, 1, .float32);
        const gate = blocks.MessageGate{
            .down = try outIn(&g, try normal(&g, &.{ 2 * h, 64 }, 10 * n + 3, 0.05, dt)),
            .forget = try outIn(&g, try normal(&g, &.{ 64, h }, 10 * n + 4, 0.2, .float32)),
            .message = try outIn(&g, try normal(&g, &.{ 64, h }, 10 * n + 5, 0.2, .float32)),
        };
        for (0..4) |j| {
            const got = (try fused(&g, hidden, messages, @intCast(j), gate.down, gate.forget, gate.message)) orelse return error.KernelDeclined;
            const want = try chain(&cpu, &gate, hidden, try blocks.row(&cpu, messages, @intCast(j)));
            try testing.expectEqualSlices(c_int, ops.shape(want), ops.shape(got));
            const off = try cpu.abs(try cpu.sub(got, want));
            const worst = try cpu.call(mlx.mlx_max, .{ off, false });
            const mean = try cpu.call(mlx.mlx_mean, .{ off, false });
            try ops.evalMany(&.{ worst, mean });
            var w: f32 = 0;
            var m: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&w, worst));
            try mlx.check(mlx.mlx_array_item_float32(&m, mean));
            try testing.expect(m <= 1e-4 and w <= 0.1);
        }
    };
    const hidden = try normal(&g, &.{ 1, MAX_ROWS + 1, h }, 1, 1, .bfloat16);
    const messages = try normal(&g, &.{ 1, MAX_ROWS + 1, 4, h }, 2, 1, .float32);
    try testing.expectEqual(@as(?Array, null), try fused(&g, hidden, messages, 0, try normal(&g, &.{ 64, 2 * h }, 3, 1, .bfloat16), try normal(&g, &.{ h, 64 }, 4, 1, .float32), try normal(&g, &.{ h, 64 }, 5, 1, .float32)));
}
