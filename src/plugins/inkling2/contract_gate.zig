//! Kernel contract registrations for the loop message gate's two kernels at the pack's width (2304, rank 64, four
//! messages): `inkling2_message_gate` (the projection of the RMS-normed pair) and `inkling2_message_gate_mix`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const gate = @import("message_gate.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const R = 64;
const NQ = 4;
const TG = gate.TG;

const proj_launch = metal.Launch{
    .name = "inkling2_message_gate",
    .inputs = &gate.PROJ_INPUTS,
    .outputs = &.{"st"},
    .source = gate.PROJ_SOURCE,
    .header = gate.PROJ_HEADER,
    .tg_bytes = 4 * TG / 32,
    .m1 = 1024,
    .m2 = 1024,
};

const mix_launch = metal.Launch{
    .name = "inkling2_message_gate_mix",
    .inputs = &gate.MIX_INPUTS,
    .outputs = &.{"out"},
    .source = gate.MIX_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// The inputs both kernels read: a shape's rows are the token rows, n the message a gate takes, k the width.
const Inputs = struct {
    hidden: []f64 = &.{},
    messages: []f64 = &.{},

    fn draw(self: *Inputs, a: std.mem.Allocator, rng: std.Random, shape: Shape, dt: mlx.mlx_dtype, fill: spec.Fill) !void {
        self.hidden = try host.uniform(a, rng, shape.rows * shape.k, -3, 3, dt);
        self.messages = try host.uniform(a, rng, shape.rows * NQ * shape.k, -2, 2, .float32);
        if (fill == .nonfinite) {
            host.plant(self.hidden, shape.rows, shape.k);
            // The gate's own message row of each token row.
            for (0..@min(shape.rows, 4)) |r| host.plant(self.messages[(r * NQ + shape.n) * shape.k ..][0..shape.k], 1, shape.k);
        }
    }

    fn message(self: *const Inputs, shape: Shape, row: usize) []const f64 {
        return self.messages[(row * NQ + shape.n) * shape.k ..][0..shape.k];
    }
};

/// The projection: `st = TD(TD(rms(hidden) | rms(message)) @ down)`.
const Proj = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    in: Inputs = .{},
    down: []f64 = &.{},
    args: [gate.PROJ_INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, dt: mlx.mlx_dtype) Proj {
        return .{ .rig = metal.Rig.init(a), .launch = l, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Proj) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Proj, shape: Shape) ?metal.Geometry {
        if (shape.rows > gate.MAX_ROWS or shape.k < TG or shape.n >= NQ) return null;
        var g = metal.Geometry{ .grid = .{ ci(shape.rows * R * TG), 1, 1 }, .tg = .{ TG, 1, 1 } };
        g.out(&.{ ci(shape.rows), R }, .float32);
        g.dtype("TD", self.dt);
        g.int("H", shape.k);
        g.int("R", R);
        g.int("NQ", NQ);
        g.int("TG", TG);
        return g;
    }

    pub fn prepare(self: *Proj, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.in.draw(a, rng, shape, self.dt, fill);
        self.down = try host.uniform(a, rng, R * 2 * shape.k, -0.1, 0.1, self.dt);
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(self.dt, &.{ 1, ci(shape.rows), ci(shape.k) }, self.in.hidden),
            try rig.input(.float32, &.{ 1, ci(shape.rows), NQ, ci(shape.k) }, self.in.messages),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(shape.n)}),
            try rig.input(self.dt, &.{ R, ci(2 * shape.k) }, self.down),
        };
        return true;
    }

    pub fn run(self: *Proj) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Proj) usize {
        return self.shape.rows * R;
    }

    /// Each normed value and each projection is rounded where the device rounds it: a value the device may hold on
    /// the other side of a rounding point carries that step as its error.
    fn emit(self: *const Proj, want: []f64, mag: ?[]f64, sim: ?struct { acc: spec.Accum, order: spec.Order }) !void {
        const a = self.rig.a;
        const h = self.shape.k;
        const x = try a.alloc(host.Val, 2 * h);
        defer a.free(x);
        const terms = try a.alloc(f64, 2 * h);
        defer a.free(terms);
        // A thread's own sum, the lanes' tree, the simdgroups' tree.
        const chain: f64 = @floatFromInt(h / TG + 5 + 3 + 1);
        for (0..self.shape.rows) |row| {
            const halves = [2][]const f64{ self.in.hidden[row * h ..][0..h], self.in.message(self.shape, row) };
            for (halves, 0..) |src, half| {
                var ssq: f64 = 0;
                for (src) |v| ssq += v * v;
                const inv = 1 / @sqrt(ssq / @as(f64, @floatFromInt(h)) + 1e-6);
                // The squares' sum, the mean, a fast rsqrt and the product.
                const rel = spec.gamma(chain, host.U) + 8 * 0x1p-23;
                for (src, x[half * h ..][0..h]) |v, *out| out.* = host.round(self.dt, .{ .v = v * inv, .e = @abs(v * inv) * rel });
            }
            for (0..R) |r| {
                var e: f64 = 0;
                var abs: f64 = 0;
                for (x, terms, 0..) |xi, *t, i| {
                    const d = self.down[r * 2 * h + i];
                    t.* = xi.v * d;
                    e += xi.e * @abs(d);
                    abs += @abs(t.*);
                }
                const sum = if (sim) |s| spec.sumIn(s.acc, s.order, terms) else host.sum(terms);
                const st = host.round(self.dt, .{ .v = sum, .e = e + spec.gamma(2 * chain, host.U) * abs });
                want[row * R + r] = st.v;
                if (mag) |m| m[row * R + r] = host.allowance(st.e);
            }
        }
    }

    pub fn reference(self: *const Proj, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Proj, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

/// The mix: `(1 - tanh(st.forget)) * hidden + tanh(st.gain) * message`.
const Mix = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    in: Inputs = .{},
    st: []f64 = &.{},
    forget: []f64 = &.{},
    gain: []f64 = &.{},
    args: [gate.MIX_INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, dt: mlx.mlx_dtype) Mix {
        return .{ .rig = metal.Rig.init(a), .launch = l, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Mix) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Mix, shape: Shape) ?metal.Geometry {
        if (shape.rows > gate.MAX_ROWS or shape.k < TG or shape.n >= NQ) return null;
        var g = metal.Geometry{ .grid = .{ ci(shape.rows * shape.k * 32), 1, 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(shape.k) }, .float32);
        g.int("H", shape.k);
        g.int("R", R);
        g.int("NQ", NQ);
        return g;
    }

    pub fn prepare(self: *Mix, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.in.draw(a, rng, shape, self.dt, fill);
        self.st = try host.uniform(a, rng, shape.rows * R, -2, 2, .float32);
        self.forget = try host.uniform(a, rng, shape.k * R, -0.1, 0.1, .float32);
        self.gain = try host.uniform(a, rng, shape.k * R, -0.1, 0.1, .float32);
        try self.bind();
        return true;
    }

    fn bind(self: *Mix) !void {
        const rig = &self.rig;
        const shape = self.shape;
        rig.reset();
        self.args = .{
            try rig.input(.float32, &.{ ci(shape.rows), R }, self.st),
            try rig.input(.float32, &.{ ci(shape.k), R }, self.forget),
            try rig.input(.float32, &.{ ci(shape.k), R }, self.gain),
            try rig.input(self.dt, &.{ 1, ci(shape.rows), ci(shape.k) }, self.in.hidden),
            try rig.input(.float32, &.{ 1, ci(shape.rows), NQ, ci(shape.k) }, self.in.messages),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(shape.n)}),
        };
    }

    pub fn run(self: *Mix) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Mix) usize {
        return self.shape.rows * self.shape.k;
    }

    fn emit(self: *const Mix, want: []f64, mag: ?[]f64, sim: ?struct { acc: spec.Accum, order: spec.Order }) void {
        const h = self.shape.k;
        var terms: [2][R]f64 = undefined;
        // Two terms a lane, the lanes' tree; a fast tanh is held to 8 f32 ulps of 1.
        const sums = spec.gamma(R / 32 + 5 + 1, host.U);
        const tanh_err = 8 * 0x1p-23;
        for (0..self.shape.rows) |row| for (0..h) |d| {
            var abs = [2]f64{ 0, 0 };
            for (0..R) |k| {
                const s = self.st[row * R + k];
                terms[0][k] = s * self.forget[d * R + k];
                terms[1][k] = s * self.gain[d * R + k];
                abs[0] += @abs(terms[0][k]);
                abs[1] += @abs(terms[1][k]);
            }
            var t: [2]f64 = undefined;
            for (&t, &terms) |*v, *x| v.* = std.math.tanh(if (sim) |s| spec.sumIn(s.acc, s.order, x) else host.sum(x));
            const hv = self.in.hidden[row * h + d];
            const mv = self.in.message(self.shape, row)[d];
            const kept = (1 - t[0]) * hv;
            const heard = t[1] * mv;
            want[row * h + d] = kept + heard;
            if (mag) |m| m[row * h + d] = host.allowance((sums * abs[0] + tanh_err) * @abs(hv) + (sums * abs[1] + tanh_err) * @abs(mv) + 3 * host.U * (@abs(kept) + @abs(heard)));
        };
    }

    pub fn reference(self: *const Mix, want: []f64, mag: []f64) !void {
        self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Mix, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

fn gateSpec(comptime name: []const u8, served: []const Shape, comptime why: []const u8) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = 512, .step = 32, .min = TG } },
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = why, .exact = false },
    };
}

fn shapes(comptime list: []const [3]usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |x, i| out[i] = .{ .rows = x[0], .n = x[1], .k = x[2] };
    const final = out;
    return &final;
}

/// A decode row, a drafter's window and the widest window the wrapper takes, at each message row; a bf16 residual
/// (a layer's q, k and v inputs) and an f32 one (the MLP's, off the depth kernel).
const windows = shapes(&.{ .{ 1, 0, 2304 }, .{ 9, 2, 2304 }, .{ 16, 3, 2304 } });
const proj_why = "the projection is rounded to the weights' dtype, coarser than either half-precision sum's error";
const mix_why = "tanh flattens a sum's error where the gate saturates";

pub fn register(set: *kc.Set) !void {
    inline for (.{ .{ "bf16", mlx.mlx_dtype.bfloat16 }, .{ "f32", mlx.mlx_dtype.float32 } }) |t| {
        try set.add(Proj, .init(set.a, &proj_launch, t[1]), gateSpec("inkling2_message_gate " ++ t[0], windows, proj_why), null);
        try set.add(Mix, .init(set.a, &mix_launch, t[1]), gateSpec("inkling2_message_gate_mix (" ++ t[0] ++ " residual)", windows, mix_why), null);
    }
}

// ── tests ──

test "kernel contract inkling2 gate: the two declared launches chained are the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        var p = Proj.init(a, &proj_launch, dt);
        defer p.deinit();
        var m = Mix.init(a, &mix_launch, dt);
        defer m.deinit();
        for (windows) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            try std.testing.expect(try m.prepare(shape, .random, 8));
            // The mix on the projection's own inputs and output.
            m.in = p.in;
            const st = try host.read(a, p.rig.output(0), p.rig.stream());
            defer a.free(st);
            m.st = st;
            try m.bind();
            try m.run();
            const ours = try a.dupe(u8, m.rig.outputBytes());
            defer a.free(ours);
            var g = ops.Graph.init(a, m.rig.stream());
            defer g.deinit();
            const got = (try gate.fused(&g, m.args[3], m.args[4], @intCast(shape.n), p.args[3], m.args[1], m.args[2])) orelse return error.Declined;
            try std.testing.expectEqualSlices(u8, ours, try m.rig.bytesOf(&.{got}));
        }
    }
}
