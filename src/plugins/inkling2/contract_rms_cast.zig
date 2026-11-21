//! Kernel contract registrations for `inkling2_rms_cast`: an RMS norm in f32 times its gain, cast to bf16, at the
//! pack's hidden width (the q/k/v and MLP inputs, f32 in) and its value heads (8 of 128, bf16 in, no gain). Its f32
//! twin is the output widened, held exactly by the unit test; here it is not written (`unread`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const rms_cast = @import("rms_cast.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const launch = metal.Launch{
    .name = "inkling2_rms_cast",
    .inputs = &rms_cast.INPUTS,
    .outputs = &rms_cast.OUTPUTS,
    .source = rms_cast.SOURCE,
    .tg_bytes = 4 * 32,
    .m1 = 1024,
    .m2 = 1024,
};

const EPS = 1e-6;

/// A shape's rows are the normed rows, k their width.
const Norm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    in: mlx.mlx_dtype,
    gain: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    args: [rms_cast.INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, in: mlx.mlx_dtype, gain: bool) Norm {
        return .{ .rig = metal.Rig.init(a), .launch = l, .in = in, .gain = gain, .arena = .init(a) };
    }

    pub fn deinit(self: *Norm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Norm, shape: Shape) ?metal.Geometry {
        if (shape.k < 1 or shape.k > rms_cast.MAX_D or shape.rows < 1) return null;
        const tg: c_int = @intCast(rms_cast.threads(shape.k));
        var g = metal.Geometry{ .grid = .{ ci(shape.rows) * tg, 1, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(shape.k) }, .bfloat16);
        g.out(&.{1}, .float32);
        g.unread = 0b10;
        g.dtype("OT", .bfloat16);
        g.int("D", shape.k);
        g.flag("W", self.gain);
        g.flag("TWIN", false);
        return g;
    }

    pub fn prepare(self: *Norm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        self.x = if (fill == .stress) try host.ints(a, rng, n, -64, 64) else try host.uniform(a, rng, n, -3, 3, self.in);
        self.w = if (fill == .stress) try a.alloc(f64, shape.k) else try host.uniform(a, rng, shape.k, 0.5, 1.5, .float32);
        if (fill == .stress) @memset(self.w, 1);
        if (!self.gain) @memset(self.w, 1);
        if (fill == .nonfinite) host.plant(self.x, shape.rows, shape.k);
        const rig = &self.rig;
        rig.reset();
        self.args[0] = try rig.input(self.in, &.{ 1, ci(shape.rows), ci(shape.k) }, self.x);
        const eps = try rig.input(.float32, &.{1}, &.{host.rt(.float32, EPS)});
        self.args[1] = if (self.gain) try rig.input(.float32, &.{ci(shape.k)}, self.w) else eps;
        self.args[2] = eps;
        return true;
    }

    pub fn run(self: *Norm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Norm) usize {
        return self.shape.rows * self.shape.k;
    }

    fn emit(self: *const Norm, want: []f64, mag: ?[]f64, sim: ?struct { acc: spec.Accum, order: spec.Order }) !void {
        const d = self.shape.k;
        const sq = try self.rig.a.alloc(f64, d);
        defer self.rig.a.free(sq);
        for (0..self.shape.rows) |r| {
            const row = self.x[r * d ..][0..d];
            for (sq, row) |*q, v| q.* = if (sim != null) spec.roundTo(.f32, v * v) else v * v;
            const ss = if (sim) |s| spec.sumIn(s.acc, s.order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + host.rt(.float32, EPS));
            for (want[r * d ..][0..d], row, self.w) |*o, v, w| o.* = w * v * inv;
            // The sum's relative error reaches the output halved; every term is positive.
            if (mag) |m| for (m[r * d ..][0..d], want[r * d ..][0..d]) |*x, o| {
                x.* = @abs(o);
            };
        }
    }

    pub fn reference(self: *const Norm, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Norm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// Four terms a thread, the lanes' tree, the simdgroups' tree.
fn normDepth(shape: Shape) usize {
    return 4 + 5 + 5 + shape.k / 4096;
}

fn normSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = normDepth,
        // The mean, the epsilon, the root, the scaled value and the gain.
        .ops = 5,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 128, .step = 1, .min = 1 } },
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the bf16 output's rounding hides them" },
    };
}

const registrations = .{
    .{ "inkling2_rms_cast (f32 in, gain, 2304)", mlx.mlx_dtype.float32, true, &[_]Shape{ .{ .rows = 1, .n = 1, .k = 2304 }, .{ .rows = 9, .n = 1, .k = 2304 } } },
    .{ "inkling2_rms_cast (bf16 in, 8x128)", mlx.mlx_dtype.bfloat16, false, &[_]Shape{ .{ .rows = 8, .n = 1, .k = 128 }, .{ .rows = 72, .n = 1, .k = 128 } } },
};

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(Norm, .init(set.a, &launch, r[1], r[2]), normSpec(r[0], r[3]), null);
}

// ── tests ──

test "kernel contract inkling2 rms_cast: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Norm = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const ours = try std.testing.allocator.dupe(u8, t.rig.outputBytes());
            defer std.testing.allocator.free(ours);
            var g = ops.Graph.init(std.testing.allocator, t.rig.stream());
            defer g.deinit();
            const got = (try rms_cast.fused(&g, t.args[0], if (t.gain) t.args[1] else null, EPS, .bfloat16, false)) orelse return error.Declined;
            try std.testing.expectEqualSlices(u8, ours, try t.rig.bytesOf(&.{got.out}));
        }
    }
}
