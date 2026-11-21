//! Inkling 2's routed experts for a decode row in two kernels: gate_up with its softcapped SwiGLU (the bank's rows
//! alternate gate and up, so one simdgroup takes both and writes the activation), then down. NVFP4 and MXFP8 banks;
//! the chain is a gather, a cast, the activation's closure and a gather.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;

const Graph = ops.Graph;
const Array = ops.Array;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/experts.metal");
pub const HEADER = @import("../../sdk/blocks/moe/gather_qmv.zig").GQMV_NVFP4_HEADER;
pub const INPUTS = [_][*:0]const u8{ "x", "w", "scales", "inds", "gscale", "caps" };
pub const OUTPUTS = [_][*:0]const u8{"y"};
/// Simdgroups a threadgroup.
pub const SGS = 8;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_experts", in_vec, out_vec, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The bank's format as the kernel reads it; null for one it does not.
pub fn mode(bank: *const Bank) ?c_int {
    if (bank.b.ctx != null or bank.s.ctx == null) return null;
    return switch (bank.qp.mode) {
        .nvfp4 => if (bank.qp.bits == 4 and bank.qp.group_size == 16) 0 else null,
        .mxfp8 => if (bank.qp.bits == 8 and bank.qp.group_size == 32) 1 else null,
        else => null,
    };
}

fn run(g: *Graph, x: Array, bank: *const Bank, inds: Array, gscale: ?Array, caps: ?[2]f32, out_rows: c_int) !?Array {
    if (!mlx.streamIsGpu(g.s)) return null;
    const m = mode(bank) orelse return null;
    const ws = ops.shape(bank.w);
    if (ws.len != 3) return null;
    const n = ws[1];
    const k = @divExact(ws[2] * 32, @as(c_int, if (m == 0) 4 else 8));
    const topk = ops.shape(inds)[0];
    const xd = ops.dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (ops.dtype(inds) != .uint32 or ops.shape(inds).len != 1) return null;
    if (@rem(k, 32) != 0) return null;
    if (gscale) |s| if (ops.dtype(s) != .float32) return null;
    const kern = getKernel() catch return null;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ topk, out_rows }, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, out_rows, topk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MODE", m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "PAIR", caps != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "GS", gscale != null));

    const c = caps orelse .{ 1, 1 };
    const caps_arr = try g.fromSlice(f32, &c, &.{2});
    // Slots it does not read bind the caps.
    const inputs = [_]Array{ x, bank.w, bank.s, inds, gscale orelse caps_arr, caps_arr };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kern, in_vec, cfg, g.s));
    var y = mlx.mlx_array_new();
    const rc = mlx.mlx_vector_array_get(&y, out_vec, 0);
    const kept = try g.adopt(y);
    try mlx.check(rc);
    return kept;
}

/// One token `x` `[K]` through its top-k experts' `gate_up` (rows alternating gate and up), each dot rounded to x's
/// dtype, times the expert's `gscale` (null = 1), through the softcapped SwiGLU: `[TOPK, N / 2]`. Null where it
/// declines.
pub fn gateUp(g: *Graph, x: Array, bank: *const Bank, inds: Array, gscale: ?Array, caps: [2]f32) !?Array {
    const n = ops.shape(bank.w)[1];
    if (@rem(n, 2 * SGS) != 0) return null;
    return run(g, x, bank, inds, gscale, caps, @divExact(n, 2));
}

/// Each slot's activation `x` `[TOPK, K]` through its expert's `down`: `[TOPK, N]` in x's dtype. Null where it
/// declines.
pub fn down(g: *Graph, x: Array, bank: *const Bank, inds: Array) !?Array {
    const n = ops.shape(bank.w)[1];
    if (@rem(n, SGS) != 0) return null;
    return run(g, x, bank, inds, null, null, n);
}

// ── tests ──

const testing = std.testing;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const compiled = @import("compiled.zig");

fn quantBank(g: *Graph, e: c_int, n: c_int, k: c_int, m: [:0]const u8, seed: u64) !Bank {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const wf = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ e, n, k }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 0.05), key }), .bfloat16);
    const nv = std.mem.eql(u8, m, "nvfp4");
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, wf, mlx.mlx_optional_int.some(if (nv) 16 else 32), mlx.mlx_optional_int.some(if (nv) 4 else 8), m.ptr, .{ .ctx = null }, g.s));
    var w = mlx.mlx_array_new();
    var s = mlx.mlx_array_new();
    _ = mlx.mlx_vector_array_get(&w, qv, 0);
    _ = mlx.mlx_vector_array_get(&s, qv, 1);
    const bank = Bank{ .w = try g.adopt(w), .s = try g.adopt(s), .qp = .{ .bits = if (nv) 4 else 8, .group_size = if (nv) 16 else 32, .mode = if (nv) .nvfp4 else .mxfp8 } };
    try ops.evalMany(&.{ bank.w, bank.s });
    return bank;
}

fn gather(g: *Graph, x: Array, bank: *const Bank, inds: Array) !Array {
    var y = mlx.mlx_array_new();
    try gatherExpertMm(&y, x, bank.w, bank.s, .{ .ctx = null }, .{ .ctx = null }, inds, bank.qp.bits, bank.qp.group_size, bank.qp.mode, false, g.s);
    return g.adopt(y);
}

/// The largest |a - b| over |b|'s bf16 ulp (a floor of 1e-3 for values near zero), and the mean |a - b|.
fn ulps(g: *Graph, a: Array, b: Array) ![2]f32 {
    const a32 = try g.astype(a, .float32);
    const b32 = try g.astype(b, .float32);
    const off = try g.abs(try g.sub(a32, b32));
    const ulp = try g.addScalar(try g.mulScalar(try g.abs(b32), 1.0 / 128.0), 1e-3);
    const worst = try g.call(mlx.mlx_max, .{ try g.div(off, ulp), false });
    const mean = try g.call(mlx.mlx_mean, .{ off, false });
    try ops.evalMany(&.{ worst, mean });
    var r: [2]f32 = undefined;
    try mlx.check(mlx.mlx_array_item_float32(&r[0], worst));
    try mlx.check(mlx.mlx_array_item_float32(&r[1], mean));
    return r;
}

// Bar: the kernels give MLX's gathers and the compiled activation to a bf16 ulp or two (sums in another order), at
// the pack's widths and both of its expert formats.
test "inkling2 experts: the decode kernels give the gather chain's activation and outputs, NVFP4 and MXFP8" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const E = 16;
    const H = 2304;
    const I = 1152;
    const K = 8;
    for ([_][:0]const u8{ "nvfp4", "mxfp8" }, 0..) |m, n| {
        const gu = try quantBank(&g, E, 2 * I, H, m, 10 * n + 1);
        const dn = try quantBank(&g, E, H, I, m, 10 * n + 2);
        const key = try g.call(mlx.mlx_random_key, .{10 * n + 3});
        const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, H }, @as(usize, 2), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key }), .bfloat16);
        const inds = try g.fromSlice(u32, &.{ 3, 0, 15, 7, 8, 1, 12, 5 }, &.{K});
        const gscale = try g.fromSlice(f32, &.{ 0.5, 1, 2, 0.25, 1.5, 3, 0.75, 1, 1, 2, 0.5, 1, 4, 1, 1, 0.125 }, &.{E});
        for ([_]?Array{ gscale, null }) |gs| {
            const act = (try gateUp(&g, x, &gu, inds, gs, .{ 4, 25 })) orelse return error.KernelDeclined;
            const rows = try g.reshape(try g.broadcast(try g.reshape(x, &.{ 1, 1, H }), &.{ K, 1, H }), &.{ K, 1, H });
            const pairs = try g.reshape(try gather(&g, rows, &gu, inds), &.{ K, I, 2 });
            const sc: ?Array = if (gs) |s| try g.reshape(try g.take(s, inds, 0), &.{ K, 1, 1 }) else null;
            const want = try g.reshape(try compiled.softcappedSwiglu(&g, try g.sliceAxis(pairs, 2, 0, 1), try g.sliceAxis(pairs, 2, 1, 2), sc, 4, 25), &.{ K, I });
            const a = try ulps(&g, act, want);
            try testing.expect(a[0] <= 2 and a[1] <= 1e-3);
            const out = (try down(&g, want, &dn, inds)) orelse return error.KernelDeclined;
            const want_out = try g.reshape(try gather(&g, try g.reshape(want, &.{ K, 1, I }), &dn, inds), &.{ K, H });
            const o = try ulps(&g, out, want_out);
            try testing.expect(o[0] <= 2 and o[1] <= 1e-3);
        }
    }
}
