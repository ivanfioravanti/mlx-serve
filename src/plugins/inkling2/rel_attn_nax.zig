//! The TensorOps (NAX, M5-class GPUs) arm of Inkling 2's attention read: the SDK's NAX flash attention with its
//! relative-bias and per-row-sink arms on, key-range passes as `nax_attention.attention` cuts them.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const nax_attention = @import("../../sdk/blocks/attention/nax_attention.zig");
const naxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const Args = @import("rel_attn.zig").Args;

const Array = ops.Array;
const Graph = ops.Graph;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/rel_attn_nax_entry.metal");
pub const HEADER = kernelSource("kernels/attention/nax_tiles.metal") ++ kernelSource("kernels/attention/nax_attention.metal");
pub const INPUTS = [_][*:0]const u8{ "q", "k", "v", "sinks", "state", "params", "rel", "key_start" };
pub const OUTPUTS = [_][*:0]const u8{ "out", "state_out" };

/// The kernel's head width (q, k and v), key block and threads.
pub const D = 128;
const BK: c_int = 32;
const THREADS: c_int = 128;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_rel_attn_nax", in_vec, out_vec, SOURCE, HEADER, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

fn halfFloat(dt: ops.Dtype) bool {
    return dt == .bfloat16 or dt == .float16;
}

/// `[B, L, Hq, D]`; null where it declines: no NAX GPU, heads not 128 wide, q/k/v not all bf16 or all f16, a chunk
/// of 8 rows or fewer.
pub fn nax(g: *Graph, a: Args) !?Array {
    if (!mlx.streamIsGpu(g.s) or !naxAvailable()) return null;
    if (mlx.mlx_array_ndim(a.q) != 4 or mlx.mlx_array_ndim(a.k) != 4 or mlx.mlx_array_ndim(a.v) != 4) return null;
    if (mlx.mlx_array_ndim(a.rel) != 4 or mlx.mlx_array_ndim(a.sinks) != 3 or mlx.mlx_array_ndim(a.key_start) != 1) return null;
    const qs = ops.shape(a.q);
    const ks = ops.shape(a.k);
    const vs = ops.shape(a.v);
    const rs = ops.shape(a.rel);
    const ss = ops.shape(a.sinks);
    const b = qs[0];
    const h = qs[1];
    const ql = qs[2];
    const kl = ks[2];
    const dt = ops.dtype(a.q);
    if (qs[3] != D or ks[3] != D or vs[3] != D or !halfFloat(dt) or ops.dtype(a.k) != dt or ops.dtype(a.v) != dt) return null;
    if (ks[0] != b or vs[0] != b or vs[1] != ks[1] or vs[2] != kl or ks[1] == 0 or @rem(h, ks[1]) != 0) return null;
    if (ql <= 8 or kl < ql or rs[0] != b or rs[1] != ql or rs[2] != h or rs[3] == 0) return null;
    if (ss[0] != b or ss[1] != ql or ss[2] != h or ops.shape(a.key_start)[0] != b) return null;

    var c = g.scope();
    defer c.deinit();
    const rel = try c.contiguous(try c.astype(a.rel, dt));
    const sinks = try c.contiguous(try c.astype(a.sinks, .float32));
    const first = try c.contiguous(try c.astype(a.key_start, .int32));
    const window: c_int = @intCast(a.window);

    var edges_buf: [nax_attention.MAX_PASSES + 1]c_int = undefined;
    const nk = @divTrunc(kl + BK - 1, BK);
    const edges = nax_attention.passEdges(&edges_buf, b * h * @divTrunc(ql + 63, 64), nk, window);
    const wn: c_int = if (edges.len > 2) 2 else 1;
    const wm = @divExact(4, wn);
    const bq = 16 * wm;
    const nq = @divTrunc(ql + bq - 1, bq);
    const nq_aligned = @divTrunc(ql, bq);
    const nk_aligned = @divTrunc(kl, BK);
    const one = [_]c_int{1};
    var state = try c.zeros(&one, .float32);
    var out: Array = undefined;
    const kern = try getKernel();
    for (edges[0 .. edges.len - 1], edges[1..], 0..) |kb_begin, kb_end, i| {
        const last = i == edges.len - 2;
        const pdata = [16]i32{ b, h, D, ql, kl, @divExact(h, ks[1]), @bitCast(a.scale), nq, nk, nq_aligned, nk_aligned, ql - nq_aligned * bq, kl - nk_aligned * BK, kl - ql, kb_begin, kb_end };
        const params = try c.fromSlice(i32, &pdata, &.{16});

        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        if (last) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, ql, h, D }, 4, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &one, 1, .float32));
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &one, 1, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{b * h * nq * bq * (D + 2)}, 1, .float32));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nq * THREADS, h, b));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, THREADS, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        for ([_]struct { [*:0]const u8, c_int }{ .{ "WM", wm }, .{ "WN", wn }, .{ "WINDOW", window }, .{ "REL", rs[3] } }) |t|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
        for ([_]struct { [*:0]const u8, bool }{ .{ "ALIGN_Q", @rem(ql, bq) == 0 }, .{ "ALIGN_K", @rem(kl, BK) == 0 }, .{ "FIRST", i == 0 }, .{ "LAST", last } }) |t|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t[0], t[1]));

        const ins = [_]Array{ a.q, a.k, a.v, sinks, state, params, rel, first };
        const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kern, in_vec, cfg, g.s));
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&y, outs, if (last) 0 else 1));
        if (last) out = try c.adopt(y) else state = try c.adopt(y);
    }
    if (!engaged) {
        engaged = true;
        @import("../../sdk/util/log.zig").info("[attn] inkling2 NAX relative attention engaged (window={d}, passes={d})\n", .{ window, edges.len - 1 });
    }
    return try g.keep(out);
}

// ── tests ──

const testing = std.testing;
const reference = @import("rel_attn.zig").reference;

fn random(g: *Graph, rnd: std.Random, sh: []const c_int, scale: f32, dt: ops.Dtype) !Array {
    var n: usize = 1;
    for (sh) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = rnd.floatNorm(f32) * scale;
    return g.astype(try g.fromSlice(f32, buf, sh), dt);
}

// Bar: within bf16 rounding of the fp32 op chain at sliding and global shapes, ragged tiles, a cache ahead of the
// chunk, a first key past 0, distances past the bias extent, and enough keys for several passes.
test "inkling2 NAX relative attention tracks the op chain" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !naxAvailable()) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x1a2b);
    const rnd = prng.random();
    const cases = [_]struct { h: c_int, hk: c_int, l: c_int, s: c_int, r: c_int, window: u32, start: i32 }{
        .{ .h = 6, .hk = 2, .l = 100, .s = 100, .r = 64, .window = 0, .start = 0 },
        .{ .h = 6, .hk = 2, .l = 77, .s = 300, .r = 64, .window = 0, .start = 0 },
        .{ .h = 4, .hk = 4, .l = 200, .s = 333, .r = 128, .window = 128, .start = 0 },
        .{ .h = 4, .hk = 1, .l = 64, .s = 190, .r = 16, .window = 0, .start = 37 },
        .{ .h = 32, .hk = 4, .l = 1024, .s = 12300, .r = 1024, .window = 0, .start = 0 },
    };
    for (cases) |cs| {
        var g = Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        const a: Args = .{
            .q = try random(&g, rnd, &.{ 1, cs.h, cs.l, D }, 0.5, .bfloat16),
            .k = try random(&g, rnd, &.{ 1, cs.hk, cs.s, D }, 0.5, .bfloat16),
            .v = try random(&g, rnd, &.{ 1, cs.hk, cs.s, D }, 0.5, .bfloat16),
            .rel = try random(&g, rnd, &.{ 1, cs.l, cs.h, cs.r }, 1.0, .bfloat16),
            .sinks = try random(&g, rnd, &.{ 1, cs.l, cs.h }, 1.0, .bfloat16),
            .key_start = try g.fromSlice(i32, &.{cs.start}, &.{1}),
            .scale = 0.0884,
            .window = cs.window,
        };
        const got = (try nax(&g, a)) orelse return error.NaxDeclined;
        var f: Args = a;
        inline for (.{ "q", "k", "v" }) |name| @field(f, name) = try g.astype(@field(a, name), .float32);
        const want = try reference(&g, f);
        const diff = try g.max(try g.reshape(try g.abs(try g.sub(try g.astype(got, .float32), want)), &.{-1}), 0, false);
        try ops.eval(diff);
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, diff));
        try testing.expect(worst < 0.02);
    }
}
