//! Kernel contract registrations for `inkling2_l2_clip`: each D-long vector times its head's scale, shrunk to an L2
//! norm, at the pack's q (24 heads of 128, scaled), k (8 of 128, unscaled) and relative (24 of 16, scaled) widths.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const l2_clip = @import("l2_clip.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const launch = metal.Launch{
    .name = "inkling2_l2_clip",
    .inputs = &l2_clip.INPUTS,
    .outputs = &l2_clip.OUTPUTS,
    .source = l2_clip.SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const THRESHOLD = 5.0;

/// A shape's rows are token rows, n its heads (one vector each), k the vector's length.
const Clip = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    scaled: bool,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    scale: []f64 = &.{},
    args: [l2_clip.INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, scaled: bool, dt: mlx.mlx_dtype) Clip {
        return .{ .rig = metal.Rig.init(a), .launch = l, .scaled = scaled, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Clip) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Clip, shape: Shape) ?metal.Geometry {
        if (shape.k < 1 or shape.k > 1024 or shape.rows * shape.n == 0) return null;
        const at = l2_clip.grid(shape.rows * shape.n);
        var g = metal.Geometry{ .grid = .{ at[0], 1, 1 }, .tg = .{ at[1], 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(shape.n), ci(shape.k) }, self.dt);
        g.dtype("T", self.dt);
        g.int("D", shape.k);
        g.int("H", shape.n);
        g.flag("SCALED", self.scaled);
        return g;
    }

    pub fn prepare(self: *Clip, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = shape.k;
        const vectors = shape.rows * shape.n;
        if (fill == .stress) {
            // Integers whose squares sum past fp16's range and bf16's precision, exact in fp32; a unit scale.
            self.x = try host.ints(a, rng, vectors * d, -32, 32);
            self.scale = try a.alloc(f64, shape.n * d);
            @memset(self.scale, 1);
        } else {
            // Half the vectors under the threshold (left as they are), half past it (shrunk).
            self.x = try a.alloc(f64, vectors * d);
            for (0..vectors) |v| host.uniformInto(rng, self.x[v * d ..][0..d], if (v % 2 == 0) -0.2 else -2, if (v % 2 == 0) 0.2 else 2, self.dt);
            self.scale = try host.uniform(a, rng, shape.n * d, 0.5, 1.5, .bfloat16);
        }
        if (!self.scaled) @memset(self.scale, 1);
        if (fill == .nonfinite) host.plant(self.x, vectors, d);

        const rig = &self.rig;
        rig.reset();
        self.args[0] = try rig.input(self.dt, &.{ 1, ci(shape.rows), ci(shape.n), ci(d) }, self.x);
        const thr = try rig.input(.float32, &.{1}, &.{THRESHOLD});
        self.args[1] = if (self.scaled) try rig.input(.bfloat16, &.{ ci(shape.n), ci(d) }, self.scale) else thr;
        self.args[2] = thr;
        return true;
    }

    pub fn run(self: *Clip) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Clip) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    /// Each vector's sum of squares (in `sim`'s accumulator and order, or float64), then the clip.
    fn emit(self: *const Clip, want: []f64, mag: ?[]f64, sim: ?struct { acc: spec.Accum, order: spec.Order }) !void {
        const a = self.rig.a;
        const d = self.shape.k;
        const sq = try a.alloc(f64, d);
        defer a.free(sq);
        const st = try a.alloc(f64, d);
        defer a.free(st);
        for (0..self.shape.rows * self.shape.n) |v| {
            const h = v % self.shape.n;
            for (st, sq, 0..) |*s, *q, i| {
                s.* = self.x[v * d + i] * self.scale[h * d + i];
                if (sim != null) s.* = spec.roundTo(.f32, s.*);
                q.* = s.* * s.*;
            }
            const ss = if (sim) |m| spec.sumIn(m.acc, m.order, sq) else host.sum(sq);
            const r = THRESHOLD / (@sqrt(ss) + 1e-6);
            const f = if (std.math.isNan(r)) r else @min(r, 1);
            for (want[v * d ..][0..d], st) |*o, s| o.* = s * f;
            // The sum's relative error reaches the factor halved; every term is positive.
            if (mag) |m| for (m[v * d ..][0..d], want[v * d ..][0..d]) |*x, o| {
                x.* = @abs(o);
            };
        }
    }

    pub fn reference(self: *const Clip, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Clip, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Clip) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const got = (try l2_clip.fused(&g, self.args[0], if (self.scaled) self.args[1] else null, THRESHOLD)) orelse return error.Declined;
        return self.rig.bytesOf(&.{got});
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// A lane's own terms, then the simdgroup's five-level tree.
fn clipDepth(shape: Shape) usize {
    return (shape.k + 31) / 32 + 5;
}

fn clipSpec(comptime name: []const u8, out: spec.Elem, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = clipDepth,
        // The scaled value, the square root, the epsilon, the division and the product.
        .ops = 5,
        .out = out,
        .tiles = .{ .{}, .{}, .{ .size = 32, .step = 1, .min = 1 } },
        .served = served,
        // A bf16 output's rounding hides an fp16 or bf16 sum's; the f32 registration proves the fp32 sums.
        .sums = if (out == .f32) .{} else .{ .f16 = 0, .bf16 = 0, .why = "the bf16 output's rounding hides them" },
    };
}

/// A decode row and a short chunk at each width the pack clips.
const registrations = .{
    .{ "inkling2_l2_clip (q, scaled 24x128)", true, mlx.mlx_dtype.bfloat16, &[_]Shape{ .{ .rows = 1, .n = 24, .k = 128 }, .{ .rows = 9, .n = 24, .k = 128 } } },
    .{ "inkling2_l2_clip (k, 8x128)", false, mlx.mlx_dtype.bfloat16, &[_]Shape{ .{ .rows = 1, .n = 8, .k = 128 }, .{ .rows = 9, .n = 8, .k = 128 } } },
    .{ "inkling2_l2_clip (rel, scaled 24x16)", true, mlx.mlx_dtype.bfloat16, &[_]Shape{ .{ .rows = 1, .n = 24, .k = 16 }, .{ .rows = 9, .n = 24, .k = 16 } } },
    .{ "inkling2_l2_clip (f32, scaled 3x128)", true, mlx.mlx_dtype.float32, &[_]Shape{ .{ .rows = 4, .n = 3, .k = 128 } } },
};

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(Clip, .init(set.a, &launch, r[1], r[2]), clipSpec(r[0], host.elemOf(r[2]), r[3]), null);
}

// ── tests ──

test "kernel contract inkling2 l2_clip: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Clip = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const ours = try std.testing.allocator.dupe(u8, t.rig.outputBytes());
            defer std.testing.allocator.free(ours);
            try std.testing.expectEqualSlices(u8, ours, try t.wrapper());
        }
    }
}
