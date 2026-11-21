//! Inkling 2's short convs in one dispatch: the op chain is a cast, the window's concat, a cast, the conv and the
//! residual's closure (five dependent dispatches), three times on a layer visit's serial path.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Graph = ops.Graph;
const Array = ops.Array;

pub const SOURCE = kernelSource("plugins/inkling2/kernels/sconv.metal");
pub const INPUTS = [_][*:0]const u8{ "x", "prev", "w", "dyn", "wts", "scale", "post" };
pub const OUTPUTS = [_][*:0]const u8{ "out", "win" };
pub const TG = 256;
/// Rows it takes (a decode row, a verify window); a prefill chunk keeps MLX's conv.
pub const MAX_ROWS = 16;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // Strided inputs pass as they are: the window's state is a slice of the visit's joined state.
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_sconv", in_vec, out_vec, SOURCE, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The dynamic form: `dyn` `[1, L, heads * K]`, the taps' per-row part; `w` is then the static `[K, C]`.
pub const Dynamic = struct { dyn: Array, heads: c_int };

pub const Out = struct { out: Array, win: Array };

/// A norm over each `hd`-wide head of the rounded output, rounded again: v's gain-less RMS, k's L2 clip.
pub const Post = union(enum) { none, rms: struct { hd: c_int, eps: f32 }, clip: struct { hd: c_int, threshold: f32 } };

/// The conv of `x` `[1, L, C]` with taps `w` (`[C, K, 1]`, or `[K, C]` with `dynamic`) after the window `prev`
/// (`[1, K - 1, C]` f32, null = zeros), its residual added and scaled by `scale`: `out` in x's dtype and the window
/// `[1, K - 1 + L, C]` f32. Null where it declines: off Metal, past `MAX_ROWS` rows, a shape or dtype it does not read.
pub fn fused(g: *Graph, x: Array, w: Array, prev: ?Array, dynamic: ?Dynamic, scale: f32) !?Out {
    return fusedSum(g, x, null, w, prev, dynamic, scale, .none);
}

/// `fused` on the weighted f32 sum of `x` `[L, KE, C]` with `weights` `[L, KE]` f32, rounded to x's dtype (null: `x`
/// is the `[1, L, C]` chunk itself).
pub fn fusedSum(g: *Graph, x: Array, weights: ?Array, w: Array, prev: ?Array, dynamic: ?Dynamic, scale: f32, post: Post) !?Out {
    if (!mlx.streamIsGpu(g.s)) return null;
    const xs = ops.shape(x);
    if (xs.len != 3) return null;
    const ke: c_int = if (weights != null) xs[1] else 0;
    if (weights) |wt| {
        if (ops.dtype(wt) != .float32 or !std.mem.eql(c_int, ops.shape(wt), &.{ xs[0], xs[1] })) return null;
    } else if (xs[0] != 1) return null;
    const sh = [_]c_int{ 1, if (weights != null) xs[0] else xs[1], xs[2] };
    const l = sh[1];
    const c = sh[2];
    const ws = ops.shape(w);
    const k = if (dynamic != null) ws[0] else ws[1];
    if (dynamic) |d| {
        if (ws.len != 2 or ws[1] != c or d.heads < 1 or @rem(c, d.heads) != 0) return null;
        const ds = ops.shape(d.dyn);
        if (ds.len != 3 or ds[0] != 1 or ds[1] != l or ds[2] != d.heads * k) return null;
    } else if (ws.len != 3 or ws[0] != c or ws[2] != 1) return null;
    if (k < 1 or l < 1 or l > MAX_ROWS or c < 1) return null;
    const post_hd: c_int = switch (post) {
        .none => 0,
        .rms => |p| p.hd,
        .clip => |p| p.hd,
    };
    if (post_hd != 0 and (@rem(post_hd, 32) != 0 or post_hd > 1024 or @rem(c, post_hd) != 0)) return null;
    if (post_hd != 0) if (dynamic) |d| if (@divExact(c, d.heads) != post_hd) return null;
    if (prev) |p| if (ops.dtype(p) != .float32 or !std.mem.eql(c_int, ops.shape(p), &.{ 1, k - 1, c })) return null;
    const dt = ops.dtype(x);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    const kern = getKernel() catch return null;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &sh, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k - 1 + l, c }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, l * c, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, if (post_hd != 0) post_hd else @min(TG, l * c), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TW", ops.dtype(w)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "C", c));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KE", ke));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HD", if (dynamic) |d| @divExact(c, d.heads) else if (post_hd != 0) post_hd else 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "POST", switch (post) {
        .none => 0,
        .rms => 1,
        .clip => 2,
    }));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "PREV", prev != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "DYN", dynamic != null));

    // Slots the kernel does not read bind x.
    const post_v: f32 = switch (post) {
        .none => 0,
        .rms => |p| p.eps,
        .clip => |p| p.threshold,
    };
    const inputs = [_]Array{ x, prev orelse x, w, if (dynamic) |d| d.dyn else x, weights orelse x, try g.fromSlice(f32, &.{scale}, &.{1}), try g.fromSlice(f32, &.{post_v}, &.{1}) };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kern, in_vec, cfg, g.s));
    var res: [2]Array = undefined;
    for (&res, 0..) |*r, i| {
        var a = mlx.mlx_array_new();
        const rc = mlx.mlx_vector_array_get(&a, out_vec, i);
        r.* = try g.adopt(a);
        try mlx.check(rc);
    }
    return .{ .out = res[0], .win = res[1] };
}

// ── tests ──

const testing = std.testing;
const roundTo = @import("../../sdk/testing/kernel_contract/spec.zig").roundTo;

fn hostOf(g: *Graph, x: Array) ![]f32 {
    const y = try g.contiguous(try g.astype(x, .float32));
    try ops.eval(y);
    var n: usize = 1;
    for (ops.shape(y)) |d| n *= @intCast(d);
    return try testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(y) orelse return error.NoData)[0..n]);
}

fn bf(v: f32) f32 {
    return @floatCast(roundTo(.bf16, v));
}

// Bar: the kernel's window equals the chain's bytes and its output is the f32 reference with the chain's bf16
// roundings (the window into the conv, the conv's sum, the output), to one bf16 ulp for a sum the kernel orders
// differently.
test "inkling2 sconv: the kernel gives the chain's window and output, static and dynamic, with and without a state" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const K = 4;
    const C = 64;
    const heads = 2;
    for ([_]c_int{ 1, 7 }) |l| for ([_]bool{ false, true }) |dyn_form| for ([_]bool{ false, true }) |with_prev| {
        const key: u64 = @as(u64, @intCast(l)) * 40 + @as(u64, @intFromBool(dyn_form)) * 20 + @as(u64, @intFromBool(with_prev)) * 10;
        const rnd = struct {
            fn f(gr: *Graph, base: u64, i: u64, sh: []const c_int, dt: ops.Dtype) !Array {
                const ki = try gr.call(mlx.mlx_random_key, .{base + i});
                const v = try gr.astype(try gr.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 1), ki }), dt);
                try ops.eval(v);
                return v;
            }
        }.f;
        const x = try rnd(&g, key, 0, &.{ 1, l, C }, .bfloat16);
        // The state as the engine holds it: a strided slice of a wider f32 array.
        const wide = try rnd(&g, key, 1, &.{ 1, K - 1, 3 * C }, .float32);
        const prev: ?Array = if (with_prev) try g.sliceAxis(wide, 2, C, 2 * C) else null;
        const w = if (dyn_form) try rnd(&g, key, 2, &.{ K, C }, .bfloat16) else try rnd(&g, key, 2, &.{ C, K, 1 }, .bfloat16);
        const dyn = try rnd(&g, key, 3, &.{ 1, l, heads * K }, .bfloat16);
        const scale: f32 = 1.0 / @sqrt(@as(f32, if (dyn_form) 2 * K + 1 else K + 1));
        const got = (try fused(&g, x, w, prev, if (dyn_form) .{ .dyn = dyn, .heads = heads } else null, scale)) orelse return error.KernelDeclined;

        const hx = try hostOf(&g, x);
        defer testing.allocator.free(hx);
        const hw = try hostOf(&g, w);
        defer testing.allocator.free(hw);
        const hd = try hostOf(&g, dyn);
        defer testing.allocator.free(hd);
        const hp = if (prev) |p| try hostOf(&g, p) else try testing.allocator.alloc(f32, 0);
        defer testing.allocator.free(hp);
        const gout = try hostOf(&g, got.out);
        defer testing.allocator.free(gout);
        const gwin = try hostOf(&g, got.win);
        defer testing.allocator.free(gwin);
        const L: usize = @intCast(l);
        // The window: the state (or zeros) then x widened.
        var win: [(K - 1 + 7) * C]f32 = undefined;
        for (0..K - 1 + L) |r| for (0..C) |c| {
            win[r * C + c] = if (r >= K - 1) hx[(r - (K - 1)) * C + c] else if (with_prev) hp[r * C + c] else 0;
        };
        try testing.expectEqualSlices(f32, win[0 .. (K - 1 + L) * C], gwin);
        const hd_dim = C / heads;
        for (0..L) |t| for (0..C) |c| {
            var acc: f32 = 0;
            if (dyn_form) {
                for (0..K) |lag| acc += win[(t + K - 1 - lag) * C + c] * (hw[lag * C + c] + hd[t * heads * K + (c / hd_dim) * K + lag]);
            } else {
                for (0..K) |k| acc += bf(win[(t + k) * C + c]) * hw[c * K + k];
                acc = bf(acc);
            }
            const want = bf((acc + hx[t * C + c]) * scale);
            const ulp = @abs(want) * (1.0 / 128.0);
            try testing.expect(@abs(gout[t * C + c] - want) <= ulp);
        };
    };
}

// Bar: summing the experts in the kernel gives the bytes of the kernel run on the f32 sum rounded to bf16.
test "inkling2 sconv: the experts' weighted sum in the kernel equals the kernel on the summed chunk" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const K = 4;
    const C = 64;
    const KE = 8;
    for ([_]c_int{ 1, 3 }) |l| {
        const L: usize = @intCast(l);
        var prng = std.Random.DefaultPrng.init(@intCast(l));
        const r = prng.random();
        var parts: [3 * KE * C]f32 = undefined;
        var wts: [3 * KE]f32 = undefined;
        var taps: [C * K]f32 = undefined;
        for (&parts) |*v| v.* = bf(r.floatNorm(f32));
        for (&wts) |*v| v.* = r.float(f32);
        for (&taps) |*v| v.* = bf(r.floatNorm(f32));
        var summed: [3 * C]f32 = undefined;
        for (0..L) |t| for (0..C) |c| {
            var acc: f32 = 0;
            for (0..KE) |j| acc += parts[(t * KE + j) * C + c] * wts[t * KE + j];
            summed[t * C + c] = bf(acc);
        };
        const p = try g.astype(try g.fromSlice(f32, parts[0 .. L * KE * C], &.{ l, KE, C }), .bfloat16);
        const wt = try g.fromSlice(f32, wts[0 .. L * KE], &.{ l, KE });
        const w = try g.astype(try g.fromSlice(f32, &taps, &.{ C, K, 1 }), .bfloat16);
        const x = try g.astype(try g.fromSlice(f32, summed[0 .. L * C], &.{ 1, l, C }), .bfloat16);
        const a = (try fusedSum(&g, p, wt, w, null, null, 0.5, .none)) orelse return error.KernelDeclined;
        const b = (try fused(&g, x, w, null, null, 0.5)) orelse return error.KernelDeclined;
        inline for (.{ .{ a.out, b.out }, .{ a.win, b.win } }) |pair| {
            const ha = try hostOf(&g, pair[0]);
            defer testing.allocator.free(ha);
            const hb = try hostOf(&g, pair[1]);
            defer testing.allocator.free(hb);
            try testing.expectEqualSlices(f32, hb, ha);
        }
    }
}

// Bar: the head norms folded into the kernel give the kernel's plain output normed by the chain, to a bf16 ulp.
test "inkling2 sconv: k's clip and v's RMS norm in the kernel equal the chain on its plain output" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const blocks = @import("blocks.zig");
    const C = 1024;
    const HD = 128;
    for ([_]c_int{ 1, 5 }) |l| {
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, @intCast(l))});
        const x = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, l, C }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 2), key }), .bfloat16);
        const w = try g.astype(try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ C, 4, 1 }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 0.5), key }), .bfloat16);
        const plain = (try fusedSum(&g, x, null, w, null, null, 0.5, .none)) orelse return error.KernelDeclined;
        const p4 = try g.reshape(plain.out, &.{ 1, l, C / HD, HD });
        const want = [_]Array{ try @import("compiled.zig").l2Clip(&g, p4, null, 5), (try blocks.rmsCast(&g, p4, null, 1e-6, .bfloat16, false)).out };
        const posts = [_]Post{ .{ .clip = .{ .hd = HD, .threshold = 5 } }, .{ .rms = .{ .hd = HD, .eps = 1e-6 } } };
        for (posts, want) |p, wv| {
            const got = (try fusedSum(&g, x, null, w, null, null, 0.5, p)) orelse return error.KernelDeclined;
            const a = try g.astype(try g.reshape(got.out, &.{ 1, l, C / HD, HD }), .float32);
            const b = try g.astype(wv, .float32);
            const rel = try g.call(mlx.mlx_max, .{ try g.div(try g.abs(try g.sub(a, b)), try g.addScalar(try g.abs(b), 1e-3)), false });
            try ops.eval(rel);
            var r: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&r, rel));
            try testing.expect(r <= 1.0 / 128.0);
        }
    }
}
