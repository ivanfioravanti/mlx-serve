//! Inkling 2's attention read: causal, optionally windowed, with a hidden-conditioned relative-position bias and one
//! sink logit per query row. `attend` runs a prefill chunk through the simdgroup kernel and everything else through
//! the op chain (mlx-lm's `_attention_reference`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const log = @import("../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// q `[B,Hq,L,D]`; k, v `[B,Hkv,S,D]` whose last L rows are the queries' own; rel `[B,L,Hq,R]`, the bias at
/// distances under R; sinks `[B,L,Hq]`; key_start `[B]`, each row's first real key. Query row i sits at key
/// `S - L + i` and sees key j at distance d = that - j when `0 <= d`, `j >= key_start` and (`window == 0` or
/// `d < window`); the logit is `q.k * scale + (d < R ? rel[d] : 0)`, and the softmax runs over the visible keys and
/// the sink, whose value is zero. The result is `[B,L,Hq,D]`.
pub const Args = struct {
    q: Array,
    k: Array,
    v: Array,
    rel: Array,
    sinks: Array,
    key_start: Array,
    scale: f32,
    window: u32,
};

pub fn attend(g: *Graph, a: Args) !Array {
    if (try @import("rel_attn_nax.zig").nax(g, a)) |out| return out;
    if (try simd(g, a)) |out| return out;
    return reference(g, a);
}

/// The op chain: fp32 scores and softmax, the weights rounded to v's dtype before they meet v.
pub fn reference(g: *Graph, a: Args) !Array {
    var c = g.scope();
    defer c.deinit();
    const qs = ops.shape(a.q);
    const ks = ops.shape(a.k);
    const b = qs[0];
    const hq = qs[1];
    const l = qs[2];
    const d = qs[3];
    const hkv = ks[1];
    const s = ks[2];
    const r = ops.shape(a.rel)[3];
    const gq = @divExact(hq, hkv);

    const qpos = try c.reshape(try c.arange(@floatFromInt(s - l), @floatFromInt(s), 1, .int32), &.{ l, 1 });
    const kpos = try c.reshape(try c.arange(0, @floatFromInt(s), 1, .int32), &.{ 1, s });
    const dist = try c.sub(qpos, kpos);
    const zero = try c.fromSlice(i32, &.{0}, &.{1});
    const first = try c.reshape(try c.astype(a.key_start, .int32), &.{ b, 1, 1, 1 });
    var visible = try c.call(mlx.mlx_logical_and, .{ try c.call(mlx.mlx_greater_equal, .{ dist, zero }), try c.call(mlx.mlx_greater_equal, .{ kpos, first }) });
    if (a.window > 0) visible = try c.call(mlx.mlx_logical_and, .{ visible, try c.less(dist, try c.fromSlice(i32, &.{@intCast(a.window)}, &.{1})) });

    const reach = try c.fromSlice(i32, &.{r}, &.{1});
    const at = try c.broadcast(try c.call(mlx.mlx_clip, .{ dist, zero, try c.fromSlice(i32, &.{r - 1}, &.{1}) }), &.{ b, hq, l, s });
    const picked = try c.astype(try c.takeAlong(try c.transpose(a.rel, &.{ 0, 2, 1, 3 }), at, -1), .float32);
    const bias = try c.where(try c.less(dist, reach), picked, try c.scalar(0, .float32));
    const mask = try c.reshape(try c.where(visible, bias, try c.scalar(-std.math.inf(f32), .float32)), &.{ b, hkv, gq, l, s });

    const q = try c.reshape(try c.astype(a.q, .float32), &.{ b, hkv, gq, l, d });
    const kt = try c.transpose(try c.expandDims(try c.astype(a.k, .float32), 2), &.{ 0, 1, 2, 4, 3 });
    const logits = try c.add(try c.mulScalar(try c.matmul(q, kt), a.scale), mask);
    const sinks = try c.reshape(try c.transpose(try c.astype(a.sinks, .float32), &.{ 0, 2, 1 }), &.{ b, hkv, gq, l, 1 });
    const p = try c.sliceAxis(try c.softmax(try c.concat(&.{ logits, sinks }, -1), -1, false), 4, 0, s);
    const out = try c.matmul(try c.astype(p, ops.dtype(a.v)), try c.expandDims(a.v, 2));
    return g.keep(try c.transpose(try c.reshape(out, &.{ b, hq, l, d }), &.{ 0, 2, 1, 3 }));
}

pub const SOURCE = kernelSource("plugins/inkling2/kernels/rel_attn.metal");
pub const HEADER = attn256.ATTN256_KERNEL_HEADER;
pub const INPUTS = [_][*:0]const u8{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale", "m_in", "l_in", "o_in" };
pub const OUTPUTS = [_][*:0]const u8{ "out", "m_out", "l_out", "o_out" };

/// Query rows a threadgroup takes (8 per simdgroup) and its threads.
pub const BQ = 64;
pub const TG = BQ / 8 * 32;
/// Keys a kernel block stages; a pass boundary is a multiple of it.
pub const BK = 16;

/// A prefill tile walking the keys for a row or two loses to the op chain.
pub const MIN_ROWS = 16;
/// Threadgroup bytes Metal grants on every Apple GPU.
const TG_MEMORY = 32768;

/// The kernel's threadgroup arrays at head width `d`: a K and a V block of 2-byte elements.
pub fn tgBytes(d: usize) usize {
    return 2 * 2 * BK * (d + 8);
}

/// Row-key pairs one dispatch may score. Past it the key axis is cut into passes that carry each row's fp32
/// state, so no single dispatch holds the GPU long enough for macOS to preempt it.
pub const DISPATCH_BUDGET: i64 = 250_000_000;
pub var budget_override: ?i64 = null;
/// Dispatches the last `simd` call issued.
pub var last_dispatches: u32 = 0;

/// Keys per pass: the most whole blocks within `budget` row-key pairs (0: all of them).
pub fn passKeys(rows: i64, s: i64, budget: i64) i64 {
    if (budget <= 0) return s;
    return @min(s, @max(BK, @divTrunc(@divTrunc(budget, @max(rows, 1)), BK) * BK));
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(&INPUTS, INPUTS.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&OUTPUTS, OUTPUTS.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K and V are cache views: read through their strides, never copied.
    const k = mlx.mlx_fast_metal_kernel_new("inkling2_rel_attn", in_vec, out_vec, SOURCE, HEADER, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

fn halfFloat(dt: ops.Dtype) bool {
    return dt == .bfloat16 or dt == .float16;
}

/// What every launch of the kernel needs of its inputs, whatever the rows: on Metal, q/k/v all bf16 or all f16, a
/// head width that is a multiple of 8 and whose tiles fit threadgroup memory, a K and V base its 16-byte loads read.
fn accepts(g: *const Graph, a: Args) bool {
    if (!mlx.streamIsGpu(g.s)) return false;
    if (mlx.mlx_array_ndim(a.q) != 4 or mlx.mlx_array_ndim(a.k) != 4 or mlx.mlx_array_ndim(a.v) != 4) return false;
    if (mlx.mlx_array_ndim(a.rel) != 4 or mlx.mlx_array_ndim(a.sinks) != 3 or mlx.mlx_array_ndim(a.key_start) != 1) return false;
    const qs = ops.shape(a.q);
    const ks = ops.shape(a.k);
    const rs = ops.shape(a.rel);
    const dt = ops.dtype(a.q);
    if (!halfFloat(dt) or ops.dtype(a.k) != dt or ops.dtype(a.v) != dt) return false;
    if (!halfFloat(ops.dtype(a.rel)) and ops.dtype(a.rel) != .float32) return false;
    if (qs[2] < 1 or ks[2] < qs[2] or qs[3] < 8 or @rem(qs[3], 8) != 0 or tgBytes(@intCast(qs[3])) > TG_MEMORY) return false;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0 or rs[3] < 1) return false;
    if (!std.mem.eql(c_int, ks, ops.shape(a.v)) or ks[0] != qs[0] or ks[3] != qs[3]) return false;
    if (rs[0] != qs[0] or rs[1] != qs[2] or rs[2] != qs[1]) return false;
    if (!std.mem.eql(c_int, ops.shape(a.sinks), rs[0..3]) or ops.shape(a.key_start)[0] != qs[0]) return false;
    return attn256.attn256Base16Aligned(a.k) and attn256.attn256Base16Aligned(a.v);
}

fn templates(cfg: mlx.mlx_fast_metal_kernel_config, a: Args) !void {
    const qs = ops.shape(a.q);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", ops.dtype(a.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", qs[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HQ", qs[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HKV", ops.shape(a.k)[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", ops.shape(a.rel)[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WINDOW", @intCast(a.window)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BQ", BQ));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", TG));
}

fn apply(c: *Graph, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Array, comptime n: usize) ![n]Array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, c.s));
    if (mlx.mlx_vector_array_size(out_vec) != n) return error.MetalKernelBadOutputCount;
    var got: [n]Array = undefined;
    for (&got, 0..) |*x, i| {
        var h = mlx.mlx_array_new();
        const rc = mlx.mlx_vector_array_get(&h, out_vec, i);
        x.* = try c.adopt(h);
        try mlx.check(rc);
    }
    return got;
}

pub const MERGE_SOURCE = kernelSource("plugins/inkling2/kernels/rel_attn_merge.metal");
pub const MERGE_INPUTS = [_][*:0]const u8{ "ms", "ls", "os" };
/// Key chunks a split launch makes at most, and the fewest keys in one.
pub const MAX_CHUNKS = 64;
pub const MIN_CHUNK = 1024;

var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// A chunk's keys for `s` keys: `MIN_CHUNK` doubled until `MAX_CHUNKS` chunks hold them. A row's chunks are cut at
/// the same keys whether it is read alone or in a window, so its bits are the same.
pub fn chunkKeys(s: i64) i64 {
    var n: i64 = MIN_CHUNK;
    while (n * MAX_CHUNKS < s) n *= 2;
    return n;
}

/// A few rows over many keys: every key chunk is read by its own threadgroups and a second kernel joins them, so
/// the dispatch is as long as one chunk, not the key range. Null where the kernel declines (`accepts`).
pub fn split(g: *Graph, a: Args) !?Array {
    if (!accepts(g, a)) return null;
    const k = getKernel() catch return null;
    if (merge_kernel == null) {
        const in_vec = mlx.mlx_vector_string_new_data(&MERGE_INPUTS, MERGE_INPUTS.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const m = mlx.mlx_fast_metal_kernel_new("inkling2_rel_attn_merge", in_vec, out_vec, MERGE_SOURCE, "", true, false);
        if (m.ctx == null) return null;
        merge_kernel = m;
    }
    const qs = ops.shape(a.q);
    const b = qs[0];
    const hq = qs[1];
    const l = qs[2];
    const d = qs[3];
    const s = ops.shape(a.k)[2];
    const chunk: c_int = @intCast(chunkKeys(s));
    const chunks = @divTrunc(s + chunk - 1, chunk);

    var c = g.scope();
    defer c.deinit();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, ops.dtype(a.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ chunks, b * l * hq }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ chunks, b * l * hq }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ chunks, b * l * hq, d }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(l + BQ - 1, BQ) * chunks * TG, hq, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TG, 1, 1));
    try templates(cfg, a);
    const dummy = try c.fromSlice(f32, &.{0}, &.{1});
    const state = try apply(&c, k, cfg, &.{
        a.q,                                                  a.k,                                               a.v,
        try c.contiguous(a.rel),                              try c.contiguous(try c.astype(a.sinks, .float32)), try c.contiguous(try c.astype(a.key_start, .int32)),
        try c.fromSlice(i32, &.{ l, s, chunk, 0, 4 }, &.{5}), try c.fromSlice(f32, &.{a.scale}, &.{1}),          dummy,
        dummy,                                                dummy,
    }, OUTPUTS.len);

    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, &[_]c_int{ b, l, hq, d }, 4, ops.dtype(a.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, b * l * hq * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "T", ops.dtype(a.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "CH", chunks));
    const out = try apply(&c, merge_kernel.?, mc, state[1..], 1);
    return try g.keep(out[0]);
}

/// The simdgroup-matrix kernel over a prefill chunk; null where it declines: under `MIN_ROWS` rows, or `accepts`.
pub fn simd(g: *Graph, a: Args) !?Array {
    if (ops.shape(a.q).len != 4 or ops.shape(a.q)[2] < MIN_ROWS or !accepts(g, a)) return null;
    const qs = ops.shape(a.q);
    const b = qs[0];
    const hq = qs[1];
    const l = qs[2];
    const d = qs[3];
    const s = ops.shape(a.k)[2];
    const dt = ops.dtype(a.q);
    const k = getKernel() catch return null;

    var c = g.scope();
    defer c.deinit();
    const rel = try c.contiguous(a.rel);
    const sinks = try c.contiguous(try c.astype(a.sinks, .float32));
    const key_start = try c.contiguous(try c.astype(a.key_start, .int32));
    const scale = try c.fromSlice(f32, &.{a.scale}, &.{1});
    const dummy = try c.fromSlice(f32, &.{0}, &.{1});

    // A window bounds a tile's keys by itself; an open layer reads every key, so its key axis takes passes.
    const rows = @as(i64, b) * hq * l;
    const step: i64 = if (a.window > 0) s else passKeys(rows, s, budget_override orelse DISPATCH_BUDGET);
    var carry: ?[3]Array = null;
    var out: Array = undefined;
    var k0: i64 = 0;
    var n: u32 = 0;
    while (k0 < s) : (k0 += step) {
        const k1 = @min(k0 + step, s);
        const last = k1 == s;
        const phase: i32 = @as(i32, @intFromBool(carry != null)) | (@as(i32, @intFromBool(last)) << 1);
        const sizes = try c.fromSlice(i32, &.{ l, s, @intCast(k0), @intCast(k1), phase }, &.{5});

        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const one = [_]c_int{1};
        const state = [_]c_int{ b, l, hq, d };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, if (last) &state else &one, if (last) 4 else 1, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, if (last) &one else &state, if (last) 1 else 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, if (last) &one else &state, if (last) 1 else 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, if (last) &one else &state, if (last) 1 else 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(l + BQ - 1, BQ) * TG, hq, b));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TG, 1, 1));
        try templates(cfg, a);

        const st = carry orelse .{ dummy, dummy, dummy };
        const got = try apply(&c, k, cfg, &.{ a.q, a.k, a.v, rel, sinks, key_start, sizes, scale, st[0], st[1], st[2] }, OUTPUTS.len);
        // The state a pass took is dead once the next pass is built.
        if (carry) |old| for (old) |x| c.drop(x);
        carry = .{ got[1], got[2], got[3] };
        out = got[0];
        n += 1;
    }
    last_dispatches = n;
    if (!engaged) {
        engaged = true;
        log.info("[inkling2] rel_attn simd engaged: L={d} S={d} window={d} passes={d}\n", .{ l, s, a.window, n });
    }
    return try g.keep(out);
}

// ── tests ──

const testing = std.testing;

const Case = struct {
    b: c_int = 1,
    hq: c_int = 4,
    hkv: c_int = 2,
    l: c_int,
    s: c_int,
    d: c_int = 32,
    r: c_int,
    window: u32 = 0,
    /// One per batch row.
    key_start: []const i32 = &.{0},
    dt: ops.Dtype = .bfloat16,
};

fn uniform(g: *Graph, rng: std.Random, sh: []const c_int, lo: f32, hi: f32, dt: ops.Dtype) !Array {
    var n: usize = 1;
    for (sh) |x| n *= @intCast(x);
    const data = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = lo + (hi - lo) * rng.float(f32);
    const out = try g.astype(try g.fromSlice(f32, data, sh), dt);
    try ops.eval(out);
    return out;
}

/// K and V arrive as views of a longer cache buffer, as the plugin's ring hands them over.
fn draw(g: *Graph, cs: Case, seed: u64) !Args {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const cap = cs.s + 5;
    return .{
        .q = try uniform(g, rng, &.{ cs.b, cs.hq, cs.l, cs.d }, -1, 1, cs.dt),
        .k = try g.sliceAxis(try uniform(g, rng, &.{ cs.b, cs.hkv, cap, cs.d }, -1, 1, cs.dt), 2, 0, cs.s),
        .v = try g.sliceAxis(try uniform(g, rng, &.{ cs.b, cs.hkv, cap, cs.d }, -2, 2, cs.dt), 2, 0, cs.s),
        .rel = try uniform(g, rng, &.{ cs.b, cs.l, cs.hq, cs.r }, -3, 3, cs.dt),
        .sinks = try uniform(g, rng, &.{ cs.b, cs.l, cs.hq }, -2, 4, cs.dt),
        .key_start = try g.fromSlice(i32, cs.key_start, &.{cs.b}),
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cs.d))),
        .window = cs.window,
    };
}

/// The same values in fp32 through the op chain: no rounding of the weights or the output.
fn truth(g: *Graph, a: Args) !Array {
    var w = a;
    w.q = try g.astype(a.q, .float32);
    w.k = try g.astype(a.k, .float32);
    w.v = try g.astype(a.v, .float32);
    w.rel = try g.astype(a.rel, .float32);
    return reference(g, w);
}

/// The kernel is no further from fp32 truth than one output ulp, and points the same way.
fn expectKernel(cs: Case, seed: u64) !void {
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const a = try draw(&g, cs, seed);
    const got = (try simd(&g, a)) orelse return error.KernelDeclined;
    const want = try truth(&g, a);
    try testing.expectEqualSlices(c_int, ops.shape(want), ops.shape(got));
    try testing.expectEqual(cs.dt, ops.dtype(got));
    const err = try attn256.attn256MaxDiff(got, want, g.s);
    const bar = @max(try attn256.attn256UlpBar(want, g.s), try attn256.attn256MaxDiff(try reference(&g, a), want, g.s));
    if (!(err <= bar)) {
        std.debug.print("rel_attn kernel: max |err| {e} over the bar {e} at L={d} S={d} window={d}\n", .{ err, bar, cs.l, cs.s, cs.window });
        return error.KernelWorseThanReference;
    }
    try testing.expect(try attn256.attn256Cosine(got, want, g.s) > 0.9999);
}

fn bruteForce(a: std.mem.Allocator, cs: Case, x: Args) ![]f64 {
    var g = Graph.init(a, mlx.mlx_default_cpu_stream_new());
    defer g.deinit();
    const names = .{ "q", "k", "v", "rel", "sinks" };
    var host: [names.len][]const f32 = undefined;
    inline for (names, 0..) |name, i| {
        const f = try g.contiguous(try g.astype(@field(x, name), .float32));
        try ops.eval(f);
        host[i] = mlx.mlx_array_data_float32(f).?[0..mlx.mlx_array_size(f)];
    }
    const q, const k, const v, const rel, const sinks = host;
    const b: usize = @intCast(cs.b);
    const hq: usize = @intCast(cs.hq);
    const hkv: usize = @intCast(cs.hkv);
    const l: usize = @intCast(cs.l);
    const s: usize = @intCast(cs.s);
    const d: usize = @intCast(cs.d);
    const r: usize = @intCast(cs.r);
    const out = try a.alloc(f64, b * l * hq * d);
    @memset(out, 0);
    const w = try a.alloc(f64, s);
    defer a.free(w);
    for (0..b) |bi| for (0..l) |i| for (0..hq) |h| {
        const hk = h / (hq / hkv);
        const row = (bi * l + i) * hq + h;
        const sink: f64 = sinks[row];
        var top = sink;
        for (0..s) |j| {
            w[j] = -std.math.inf(f64);
            if (j > s - l + i or j < @as(usize, @intCast(cs.key_start[bi]))) continue;
            const dist = s - l + i - j;
            if (cs.window > 0 and dist >= cs.window) continue;
            var dot: f64 = 0;
            for (0..d) |e| dot += @as(f64, q[((bi * hq + h) * l + i) * d + e]) * k[((bi * hkv + hk) * s + j) * d + e];
            w[j] = dot * x.scale + (if (dist < r) @as(f64, rel[row * r + dist]) else 0);
            top = @max(top, w[j]);
        }
        var z = @exp(sink - top);
        for (w) |*p| {
            p.* = @exp(p.* - top);
            z += p.*;
        }
        for (0..s) |j| for (0..d) |e| {
            out[row * d + e] += w[j] / z * v[((bi * hkv + hk) * s + j) * d + e];
        };
    };
    return out;
}

test "inkling2 rel_attn: the op chain is the formula, key by key" {
    const a = testing.allocator;
    const cases = [_]Case{
        .{ .l = 5, .s = 9, .d = 8, .r = 4, .window = 3, .key_start = &.{2}, .dt = .float32 },
        .{ .b = 2, .l = 6, .s = 6, .d = 8, .r = 16, .key_start = &.{ 0, 3 }, .dt = .float32 },
        .{ .l = 1, .s = 7, .d = 8, .r = 3, .dt = .float32 },
    };
    for (cases, 0..) |cs, n| {
        var g = Graph.init(a, mlx.mlx_default_cpu_stream_new());
        defer g.deinit();
        const x = try draw(&g, cs, 11 + n);
        const want = try bruteForce(a, cs, x);
        defer a.free(want);
        const got = try g.contiguous(try reference(&g, x));
        try ops.eval(got);
        try testing.expectEqualSlices(c_int, &.{ cs.b, cs.l, cs.hq, cs.d }, ops.shape(got));
        for (mlx.mlx_array_data_float32(got).?[0..want.len], want) |y, w| try testing.expectApproxEqAbs(w, @as(f64, y), 2e-5);
    }
}

test "inkling2 rel_attn: the kernel is no worse than the op chain against fp32, sliding and global" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const cases = [_]Case{
        .{ .l = 64, .s = 64, .r = 48, .window = 48 },
        .{ .l = 64, .s = 64, .r = 64 },
        // A chunk that is not whole tiles, behind a cache.
        .{ .l = 45, .s = 173, .r = 96, .window = 96 },
        .{ .l = 45, .s = 173, .r = 40 },
        // Rows whose first real key is past the buffer's start, alone and per batch row.
        .{ .l = 37, .s = 120, .r = 64, .key_start = &.{51} },
        .{ .b = 2, .l = 33, .s = 80, .r = 32, .window = 40, .key_start = &.{ 0, 29 } },
        // Distances past the bias table, in the window and in the open layer.
        .{ .l = 50, .s = 200, .r = 16, .window = 120 },
        .{ .l = 50, .s = 200, .r = 16 },
        .{ .hq = 6, .hkv = 2, .l = 40, .s = 90, .d = 128, .r = 64, .window = 64 },
        .{ .l = 40, .s = 90, .r = 64, .dt = .float16 },
    };
    for (cases, 0..) |cs, n| try expectKernel(cs, 100 + n);
}

test "inkling2 rel_attn: a few rows over key chunks are no worse than the op chain, and a row alone has its window's bits" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const cases = [_]Case{
        // Three chunks, the last one ragged; one chunk; a window; a first key inside the second chunk.
        .{ .l = 6, .s = 2 * MIN_CHUNK + 300, .r = 64 },
        .{ .l = 1, .s = 200, .r = 64 },
        .{ .l = 16, .s = 3 * MIN_CHUNK, .r = 512, .window = 700 },
        .{ .hq = 6, .hkv = 2, .l = 9, .s = 2 * MIN_CHUNK + 40, .d = 128, .r = 1024, .key_start = &.{MIN_CHUNK + 17} },
    };
    for (cases, 0..) |cs, n| {
        const a = try draw(&g, cs, 300 + n);
        const got = (try split(&g, a)) orelse return error.KernelDeclined;
        const want = try truth(&g, a);
        try testing.expectEqualSlices(c_int, ops.shape(want), ops.shape(got));
        const err = try attn256.attn256MaxDiff(got, want, g.s);
        const bar = @max(try attn256.attn256UlpBar(want, g.s), try attn256.attn256MaxDiff(try reference(&g, a), want, g.s));
        try testing.expect(err <= bar);
        try testing.expect(try attn256.attn256Cosine(got, want, g.s) > 0.9999);

        // The last row read alone over the same keys.
        var one = a;
        one.q = try g.sliceAxis(a.q, 2, cs.l - 1, cs.l);
        one.rel = try g.sliceAxis(a.rel, 1, cs.l - 1, cs.l);
        one.sinks = try g.sliceAxis(a.sinks, 1, cs.l - 1, cs.l);
        try ops.expectBitEqual((try split(&g, one)).?, try g.sliceAxis(got, 1, cs.l - 1, cs.l));
    }
    try testing.expectEqual(@as(i64, MIN_CHUNK), chunkKeys(MIN_CHUNK * MAX_CHUNKS));
    try testing.expectEqual(@as(i64, 2 * MIN_CHUNK), chunkKeys(MIN_CHUNK * MAX_CHUNKS + 1));
}

test "inkling2 rel_attn: key passes carry the state to the same bits as one dispatch" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const cs = Case{ .l = 45, .s = 300, .r = 64, .key_start = &.{7} };
    const a = try draw(&g, cs, 5);
    defer budget_override = null;
    budget_override = 0;
    const whole = (try simd(&g, a)).?;
    try testing.expectEqual(@as(u32, 1), last_dispatches);
    budget_override = @as(i64, cs.hq) * cs.l * 4 * BK;
    const passes = (try simd(&g, a)).?;
    try testing.expectEqual(@as(u32, 5), last_dispatches);
    try ops.expectBitEqual(passes, whole);
}

test "inkling2 rel_attn: attend takes the kernel for a prefill chunk and the op chain where it declines" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const chunk = try draw(&g, .{ .l = 32, .s = 60, .r = 16 }, 1);
    try ops.expectBitEqual(try attend(&g, chunk), (try simd(&g, chunk)).?);
    const declined = [_]Case{
        .{ .l = 1, .s = 60, .r = 16 },
        .{ .l = 32, .s = 60, .r = 16, .dt = .float32 },
        .{ .l = 32, .s = 60, .d = 12, .r = 16 },
        .{ .hq = 2, .hkv = 1, .l = 32, .s = 40, .d = 1024, .r = 16 },
    };
    for (declined, 0..) |cs, n| {
        const a = try draw(&g, cs, 20 + n);
        try testing.expectEqual(@as(?Array, null), try simd(&g, a));
        try ops.expectBitEqual(try attend(&g, a), try reference(&g, a));
    }
    // A K view that starts 6 bytes into its buffer.
    var odd = chunk;
    var prng = std.Random.DefaultPrng.init(9);
    odd.k = try g.sliceAxis(try uniform(&g, prng.random(), &.{ 1, 2, 60, 40 }, -1, 1, .bfloat16), 3, 3, 35);
    try ops.eval(odd.k);
    try testing.expectEqual(@as(?Array, null), try simd(&g, odd));
}

test "inkling2 rel_attn: a pass takes whole key blocks within the budget, one at least" {
    try testing.expectEqual(@as(i64, 5072), passKeys(24 * 2048, 65536, DISPATCH_BUDGET));
    try testing.expectEqual(@as(i64, 4096), passKeys(24 * 2048, 4096, DISPATCH_BUDGET));
    try testing.expectEqual(@as(i64, BK), passKeys(1 << 40, 4096, DISPATCH_BUDGET));
    try testing.expectEqual(@as(i64, 4096), passKeys(24 * 2048, 4096, 0));
}
