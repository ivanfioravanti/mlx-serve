//! Kernel contract registrations for `inkling2_sconv`: the short causal conv (static depthwise taps on the sliding
//! layers' k and v and every layer's post-attention and post-MLP convs, hidden-conditioned per-head taps on the global
//! layers' k and v) with its window, residual and scale, at the pack's widths and a decode row or a verify window.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const sconv = @import("sconv.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const launch = metal.Launch{
    .name = "inkling2_sconv",
    .inputs = &sconv.INPUTS,
    .outputs = &sconv.OUTPUTS,
    .source = sconv.SOURCE,
    .contiguous = false,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const K = 4;
const POST_HD = 128;
const CLIP = 5.0;
const EPS = 1e-6;
/// The dynamic form's heads (the pack's 8 KV heads of 128).
const HEADS = 8;

/// A shape's rows are the chunk's rows, n its channels, k the taps.
const Conv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dynamic: bool,
    /// Experts summed into the chunk (0: the chunk is x itself).
    experts: usize,
    /// The head norm after the conv: 0 none, 1 RMS (v), 2 L2 clip (k), over 128-wide heads.
    post: u8 = 0,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    /// With `experts`: their outputs `[L, KE, C]` and weights `[L, KE]`, whose sum is `x`.
    parts: []f64 = &.{},
    wts: []f64 = &.{},
    prev: []f64 = &.{},
    w: []f64 = &.{},
    dyn: []f64 = &.{},
    scale: f64 = 0,
    args: [sconv.INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, dynamic: bool, experts: usize) Conv {
        return .{ .rig = metal.Rig.init(a), .launch = l, .dynamic = dynamic, .experts = experts, .arena = .init(a) };
    }

    pub fn deinit(self: *Conv) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Conv, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > sconv.MAX_ROWS or shape.k != K or (self.dynamic and shape.n % HEADS != 0)) return null;
        if (self.post != 0 and shape.n % POST_HD != 0) return null;
        const threads = shape.rows * shape.n;
        var g = metal.Geometry{ .grid = .{ ci(threads), 1, 1 }, .tg = .{ ci(if (self.post != 0) POST_HD else @min(sconv.TG, threads)), 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(shape.n) }, .bfloat16);
        g.out(&.{ 1, ci(K - 1 + shape.rows), ci(shape.n) }, .float32);
        g.dtype("T", .bfloat16);
        g.dtype("TW", .bfloat16);
        g.int("C", shape.n);
        g.int("K", K);
        g.int("KE", self.experts);
        g.int("HD", if (self.dynamic) shape.n / HEADS else if (self.post != 0) @as(usize, POST_HD) else 1);
        g.int("POST", self.post);
        g.flag("PREV", true);
        g.flag("DYN", self.dynamic);
        return g;
    }

    pub fn prepare(self: *Conv, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = shape.rows;
        const c = shape.n;
        if (fill == .stress) {
            // Integer windows and taps: every product and four-term sum exact in fp32.
            self.x = try host.ints(a, rng, l * c, -8, 8);
            self.prev = try host.ints(a, rng, (K - 1) * c, -8, 8);
            self.w = try host.ints(a, rng, K * c, -4, 4);
            self.dyn = try host.ints(a, rng, l * HEADS * K, -4, 4);
        } else {
            self.x = try host.uniform(a, rng, l * c, -3, 3, .bfloat16);
            self.prev = try host.uniform(a, rng, (K - 1) * c, -3, 3, .float32);
            self.w = try host.uniform(a, rng, K * c, -1, 1, .bfloat16);
            self.dyn = try host.uniform(a, rng, l * HEADS * K, -1, 1, .bfloat16);
        }
        if (self.experts > 0) {
            // x is the experts' weighted sum, in f32 in expert order, rounded to bf16 (the kernel's own sum).
            const ke = self.experts;
            self.parts = if (fill == .stress) try host.ints(a, rng, l * ke * c, -8, 8) else try host.uniform(a, rng, l * ke * c, -3, 3, .bfloat16);
            self.wts = if (fill == .stress) try host.ints(a, rng, l * ke, 0, 1) else try host.uniform(a, rng, l * ke, 0, 1, .float32);
            if (fill == .nonfinite) host.plant(self.parts, l * ke, c);
            for (0..l) |t| for (0..c) |ch| {
                var acc: f64 = 0;
                for (0..ke) |j| acc = spec.roundTo(.f32, acc + spec.roundTo(.f32, self.parts[(t * ke + j) * c + ch] * self.wts[t * ke + j]));
                self.x[t * c + ch] = spec.roundTo(.bf16, acc);
            };
        } else if (fill == .nonfinite) host.plant(self.x, l, c);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, if (self.dynamic) 2 * K + 1 else K + 1)));

        const rig = &self.rig;
        rig.reset();
        self.args[0] = if (self.experts > 0) try rig.input(.bfloat16, &.{ ci(l), ci(self.experts), ci(c) }, self.parts) else try rig.input(.bfloat16, &.{ 1, ci(l), ci(c) }, self.x);
        self.args[1] = try rig.input(.float32, &.{ 1, K - 1, ci(c) }, self.prev);
        // Static taps `[C, K, 1]` hold w[k * C + c] at c * K + k.
        const taps = try a.alloc(f64, K * c);
        for (0..c) |ch| for (0..K) |k| {
            taps[ch * K + k] = self.w[k * c + ch];
        };
        self.args[2] = if (self.dynamic) try rig.input(.bfloat16, &.{ K, ci(c) }, self.w) else try rig.input(.bfloat16, &.{ ci(c), K, 1 }, taps);
        self.args[3] = if (self.dynamic) try rig.input(.bfloat16, &.{ 1, ci(l), HEADS * K }, self.dyn) else self.args[0];
        self.args[4] = if (self.experts > 0) try rig.input(.float32, &.{ ci(l), ci(self.experts) }, self.wts) else self.args[0];
        self.args[5] = try rig.input(.float32, &.{1}, &.{self.scale});
        self.args[6] = try rig.input(.float32, &.{1}, &.{host.rt(.float32, self.postValue())});
        return true;
    }

    pub fn run(self: *Conv) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Conv) usize {
        return self.shape.rows * self.shape.n + (K - 1 + self.shape.rows) * self.shape.n;
    }

    fn postValue(self: *const Conv) f64 {
        return if (self.post == 2) CLIP else EPS;
    }

    fn window(self: *const Conv, r: usize, ch: usize) f64 {
        return if (r < K - 1) self.prev[r * self.shape.n + ch] else self.x[(r - (K - 1)) * self.shape.n + ch];
    }

    /// The outputs, then the window. The static form rounds the window and its sum to bf16 as MLX's bf16 conv does.
    fn emit(self: *const Conv, want: []f64, mag: ?[]f64, sim: ?struct { acc: spec.Accum, order: spec.Order }) !void {
        const l = self.shape.rows;
        const c = self.shape.n;
        const hd = c / HEADS;
        var terms: [K]f64 = undefined;
        for (0..l) |t| for (0..c) |ch| {
            var m: f64 = 0;
            for (&terms, 0..) |*term, i| {
                if (self.dynamic) {
                    const tap = self.w[i * c + ch] + self.dyn[t * HEADS * K + (ch / hd) * K + i];
                    term.* = self.window(t + K - 1 - i, ch) * (if (sim != null) spec.roundTo(.f32, tap) else tap);
                } else term.* = spec.roundTo(.bf16, self.window(t + i, ch)) * self.w[i * c + ch];
                if (sim != null) term.* = spec.roundTo(.f32, term.*);
                m += @abs(term.*);
            }
            var y = if (sim) |s| spec.sumIn(s.acc, s.order, &terms) else host.sum(&terms);
            if (!self.dynamic) y = spec.roundTo(.bf16, y);
            const xt = self.x[t * c + ch];
            want[t * c + ch] = (y + xt) * self.scale;
            if (mag) |mm| mm[t * c + ch] = (if (self.dynamic) m else @abs(y)) * self.scale;
        };
        // The head norm reads the output rounded to bf16, as the kernel and the op chain round it.
        if (self.post != 0) for (0..l) |t| for (0..c / POST_HD) |h| {
            const row = want[t * c + h * POST_HD ..][0..POST_HD];
            var ss: f64 = 0;
            for (row) |*o| {
                o.* = spec.roundTo(.bf16, o.*);
                ss += o.* * o.*;
            }
            const pv = host.rt(.float32, self.postValue());
            const factor = if (self.post == 1) 1 / @sqrt(ss / POST_HD + pv) else blk: {
                const r = pv / (@sqrt(ss) + 1e-6);
                break :blk if (std.math.isNan(r)) r else @min(r, 1);
            };
            for (row, 0..) |*o, i| {
                o.* *= factor;
                if (mag) |mm| mm[t * c + h * POST_HD + i] *= @abs(factor);
            }
        };
        const at = l * c;
        for (0..K - 1 + l) |r| for (0..c) |ch| {
            want[at + r * c + ch] = self.window(r, ch);
            if (mag) |mm| mm[at + r * c + ch] = 0;
        };
    }

    pub fn reference(self: *const Conv, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Conv, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Conv) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const got = (try sconv.fusedSum(&g, self.args[0], if (self.experts > 0) self.args[4] else null, self.args[2], self.args[1], if (self.dynamic) .{ .dyn = self.args[3], .heads = HEADS } else null, @floatCast(self.scale), switch (self.post) {
            0 => .none,
            1 => .{ .rms = .{ .hd = POST_HD, .eps = EPS } },
            else => .{ .clip = .{ .hd = POST_HD, .threshold = CLIP } },
        })) orelse return error.Declined;
        return self.rig.bytesOf(&.{ got.out, got.win });
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

fn outCount(shape: Shape) usize {
    return shape.rows * shape.n;
}

fn depthK(_: Shape) usize {
    return K;
}

fn convSpec(comptime name: []const u8, comptime dynamic: bool, comptime post: u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthK,
        // The residual add and the scale; the static form's conv sum is rounded to bf16 before them.
        .ops = if (post != 0) 6 else 2,
        .mids = (if (dynamic) @as(f64, 0) else 1) + (if (post != 0) @as(f64, 2) else 0),
        .out = .bf16,
        .outs = &.{ .{ .elem = .bf16, .count = outCount }, .{ .elem = .f32, .count = outCount } },
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "four-term sums the bf16 output's rounding hides" },
    };
}

/// A decode row, a verify window and short prefill chunks at the widths the pack convolves: k and v (1024), the hidden
/// width (2304).
/// The post-MLP conv also takes the 8 experts' outputs and sums them itself.
const registrations = .{
    .{ "inkling2_sconv (static)", false, 0, &[_]Shape{ .{ .rows = 1, .n = 2304, .k = K }, .{ .rows = 9, .n = 1024, .k = K }, .{ .rows = 33, .n = 2304, .k = K }, .{ .rows = 64, .n = 1024, .k = K } } },
    .{ "inkling2_sconv (dynamic, 8 heads)", true, 0, &[_]Shape{ .{ .rows = 1, .n = 1024, .k = K }, .{ .rows = 9, .n = 1024, .k = K }, .{ .rows = 64, .n = 1024, .k = K } } },
    .{ "inkling2_sconv (static, 8 experts summed)", false, 8, &[_]Shape{ .{ .rows = 1, .n = 2304, .k = K }, .{ .rows = 9, .n = 2304, .k = K }, .{ .rows = 64, .n = 2304, .k = K } } },
};

/// k's L2 clip and v's RMS norm after the conv, over the 8 KV heads of 128.
const post_registrations = .{
    .{ "inkling2_sconv (static, k clip)", false, 2 },
    .{ "inkling2_sconv (static, v rms)", false, 1 },
    .{ "inkling2_sconv (dynamic, k clip)", true, 2 },
    .{ "inkling2_sconv (dynamic, v rms)", true, 1 },
};
const post_shapes = &[_]Shape{ .{ .rows = 1, .n = 1024, .k = K }, .{ .rows = 9, .n = 1024, .k = K }, .{ .rows = 64, .n = 1024, .k = K } };

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(Conv, .init(set.a, &launch, r[1], r[2]), convSpec(r[0], r[1], 0, r[3]), null);
    inline for (post_registrations) |r| {
        var t = Conv.init(set.a, &launch, r[1], 0);
        t.post = r[2];
        try set.add(Conv, t, convSpec(r[0], r[1], r[2], post_shapes), null);
    }
}

// ── tests ──

test "kernel contract inkling2 sconv: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Conv = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const ours = try std.testing.allocator.dupe(u8, t.rig.outputBytes());
            defer std.testing.allocator.free(ours);
            try std.testing.expectEqualSlices(u8, ours, try t.wrapper());
        }
    }
}
