//! Kernel contract registrations for `inkling2_experts`: a decode row's top-8 experts read in place, the gate_up
//! pair with its softcapped SwiGLU and the down projection, over NVFP4 and MXFP8 banks at the pack's widths
//! (hidden 2304, intermediate 1152).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const experts = @import("experts.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const launch = metal.Launch{
    .name = "inkling2_experts",
    .inputs = &experts.INPUTS,
    .outputs = &experts.OUTPUTS,
    .source = experts.SOURCE,
    .header = experts.HEADER,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// The bank's experts; the slots pick them.
const E = 12;
const CAPS = [2]f64{ 4, 25 };
const SLOTS = [8]u32{ 3, 0, 11, 7, 8, 1, 10, 5 };

/// A shape's rows are the top-k slots, n the output rows (gate_up's activations or down's outputs), k the input.
const Experts = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    mxfp8: bool,
    pair: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    /// Dequantized bank `[E, rows, k]`, its codes and scales as bound.
    w: []f64 = &.{},
    gscale: []f64 = &.{},
    args: [experts.INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, mxfp8: bool, pair: bool) Experts {
        return .{ .rig = metal.Rig.init(a), .launch = l, .mxfp8 = mxfp8, .pair = pair, .arena = .init(a) };
    }

    pub fn deinit(self: *Experts) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn bankRows(self: *const Experts, shape: Shape) usize {
        return if (self.pair) 2 * shape.n else shape.n;
    }

    pub fn geometry(self: *const Experts, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > SLOTS.len or shape.k % 32 != 0 or shape.n % experts.SGS != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(shape.n), ci(shape.rows) }, .tg = .{ 32, experts.SGS, 1 } };
        g.out(&.{ ci(shape.rows), ci(shape.n) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("MODE", @as(c_int, if (self.mxfp8) 1 else 0));
        g.int("K", shape.k);
        g.int("N", self.bankRows(shape));
        g.flag("PAIR", self.pair);
        g.flag("GS", self.pair);
        return g;
    }

    pub fn prepare(self: *Experts, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = self.bankRows(shape);
        const k = shape.k;
        const vpw: usize = if (self.mxfp8) 4 else 8;
        const gsz: usize = if (self.mxfp8) 32 else 16;
        const stress = fill == .stress;
        const words = try a.alloc(f64, E * rows * k / vpw);
        const scales = try a.alloc(f64, E * rows * k / gsz);
        self.w = try a.alloc(f64, E * rows * k);
        for (scales) |*s| s.* = if (stress) (if (self.mxfp8) 127 else 0x38) else if (self.mxfp8) @floatFromInt(120 + rng.uintLessThan(u8, 8)) else @floatFromInt(host.e4m3Code(rng, -4, 0, false));
        for (words, 0..) |*wd, i| {
            var word: u32 = 0;
            for (0..vpw) |j| {
                // Stress codes keep every product a multiple of a half and every sum exact in fp32.
                // The pair's stress codes are non-negative: its gate dot then stays where the sigmoid is relative-exact.
                const code: u32 = if (self.mxfp8)
                    (if (stress) @as(u32, host.e4m3Code(rng, 0, 2, !self.pair)) else @as(u32, host.e4m3Code(rng, -6, 2, true)))
                else
                    rng.uintLessThan(u32, if (stress and self.pair) 8 else 16);
                word |= code << @intCast(j * (32 / vpw));
                const at = i * vpw + j;
                const sc = scales[at / gsz];
                self.w[at] = if (self.mxfp8) host.e4m3(@intCast(code)) * host.e8m0(@intFromFloat(sc)) else host.e2m1(@intCast(code)) * host.e4m3(@intFromFloat(sc));
            }
            wd.* = @floatFromInt(word);
        }
        const xn = if (self.pair) k else shape.rows * k;
        self.x = if (stress) try host.ints(a, rng, xn, if (self.pair) 0 else -8, 8) else try host.uniform(a, rng, xn, -2, 2, .bfloat16);
        self.gscale = try a.alloc(f64, E);
        for (self.gscale) |*s| s.* = if (stress) 1 else host.rt(.float32, 0.25 + rng.float(f64));
        // The pair shares one x, so its non-finite values ride the expert scales: a NaN and an infinity, one slot each.
        if (fill == .nonfinite) {
            if (self.pair) {
                self.gscale[SLOTS[0]] = std.math.nan(f64);
                self.gscale[SLOTS[1]] = std.math.inf(f64);
            } else host.plant(self.x, xn / k, k);
        }

        const rig = &self.rig;
        rig.reset();
        self.args[0] = try rig.input(.bfloat16, if (self.pair) &.{ 1, ci(k) } else &.{ ci(shape.rows), ci(k) }, self.x);
        self.args[1] = try rig.input(.uint32, &.{ E, ci(rows), ci(k / vpw) }, words);
        self.args[2] = try rig.input(.uint8, &.{ E, ci(rows), ci(k / gsz) }, scales);
        var slots: [SLOTS.len]f64 = undefined;
        for (&slots, SLOTS) |*s, v| s.* = @floatFromInt(v);
        self.args[3] = try rig.input(.uint32, &.{ci(shape.rows)}, slots[0..shape.rows]);
        self.args[5] = try rig.input(.float32, &.{2}, &CAPS);
        self.args[4] = if (self.pair) try rig.input(.float32, &.{E}, self.gscale) else self.args[5];
        return true;
    }

    pub fn run(self: *Experts) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Experts) usize {
        return self.shape.rows * self.shape.n;
    }

    fn dot(self: *const Experts, eid: usize, row: usize, xs: []const f64, sim: ?Sim, mag: *f64) f64 {
        const k = self.shape.k;
        const wr = self.w[(eid * self.bankRows(self.shape) + row) * k ..][0..k];
        var terms: [4096]f64 = undefined;
        for (terms[0..k], xs, wr) |*t, x, w| {
            t.* = if (sim != null) spec.roundTo(.f32, x * w) else x * w;
            mag.* += @abs(x * w);
        }
        return if (sim) |s| spec.sumIn(s.acc, s.order, terms[0..k]) else host.sum(terms[0..k]);
    }

    const Sim = struct { acc: spec.Accum, order: spec.Order };

    fn emit(self: *const Experts, want: []f64, mag: ?[]f64, sim: ?Sim) !void {
        const k = self.shape.k;
        for (0..self.shape.rows) |e| for (0..self.shape.n) |n| {
            const eid = SLOTS[e];
            var m: f64 = 0;
            const at = e * self.shape.n + n;
            if (self.pair) {
                const s = self.gscale[eid];
                var mg: f64 = 0;
                var mu: f64 = 0;
                const gt = spec.roundTo(.bf16, self.dot(eid, 2 * n, self.x[0..k], sim, &mg)) * s;
                const ut = spec.roundTo(.bf16, self.dot(eid, 2 * n + 1, self.x[0..k], sim, &mu)) * s;
                const sig = 1 / (1 + @exp(-gt));
                const tg = std.math.tanh(gt / CAPS[0]);
                const tu = std.math.tanh(ut / CAPS[1]);
                const act = tg * CAPS[0] * sig;
                want[at] = act * tu * CAPS[1];
                // The dots' errors through the activation's slopes: their sums (scaled by gamma in the bound) and their
                // one rounding to bf16, which the kernel may take the other way at a tie (written in gamma's units).
                if (mag) |mm| {
                    const gam = spec.gamma(@floatFromInt(depthOf(self.shape)), 0x1p-24);
                    const dg = @abs(((1 - tg * tg) * sig + act * (1 - sig)) * tu * CAPS[1]);
                    const du = @abs(act * (1 - tu * tu));
                    // MLX's sigmoid takes 1 - y below zero: its error is absolute (a few 2^-24), not relative.
                    const dsig = @abs(tg * CAPS[0] * tu * CAPS[1]);
                    mm[at] = dg * (@abs(s) * mg + @abs(gt) * 0x1p-8 / gam) + du * (@abs(s) * mu + @abs(ut) * 0x1p-8 / gam) + dsig * 4 * 0x1p-24 / gam;
                }
            } else {
                want[at] = self.dot(eid, n, self.x[e * k ..][0..k], sim, &m);
                if (mag) |mm| mm[at] = m;
            }
        };
    }

    pub fn reference(self: *const Experts, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Experts, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Experts) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const bank = Bank{ .w = self.args[1], .s = self.args[2], .qp = .{ .bits = if (self.mxfp8) 8 else 4, .group_size = if (self.mxfp8) 32 else 16, .mode = if (self.mxfp8) .mxfp8 else .nvfp4 } };
        const got = if (self.pair)
            (try experts.gateUp(&g, self.args[0], &bank, self.args[3], self.args[4], .{ @floatCast(CAPS[0]), @floatCast(CAPS[1]) })) orelse return error.Declined
        else
            (try experts.down(&g, self.args[0], &bank, self.args[3])) orelse return error.Declined;
        return self.rig.bytesOf(&.{got});
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// A lane's packs, its four accumulators, the simdgroup's tree.
fn depthOf(shape: Shape) usize {
    return shape.k / 32 / 4 + 3 + 5;
}

fn expertsSpec(comptime name: []const u8, comptime pair: bool, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthOf,
        // The pair: the dots' decodes, the scale, the caps' divisions, two tanh, the sigmoid and three products.
        .ops = if (pair) 12 else 3,
        .out = .bf16,
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the bf16 output's rounding hides them" },
    };
}

const registrations = .{
    .{ "inkling2_experts (gate_up, NVFP4)", false, true, &[_]Shape{.{ .rows = 8, .n = 1152, .k = 2304 }} },
    .{ "inkling2_experts (down, NVFP4)", false, false, &[_]Shape{.{ .rows = 8, .n = 2304, .k = 1152 }} },
    .{ "inkling2_experts (gate_up, MXFP8)", true, true, &[_]Shape{.{ .rows = 8, .n = 1152, .k = 2304 }} },
    .{ "inkling2_experts (down, MXFP8)", true, false, &[_]Shape{.{ .rows = 8, .n = 2304, .k = 1152 }} },
};

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(Experts, .init(set.a, &launch, r[1], r[2]), expertsSpec(r[0], r[2], r[3]), null);
}

// ── tests ──

test "kernel contract inkling2 experts: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Experts = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const ours = try std.testing.allocator.dupe(u8, t.rig.outputBytes());
            defer std.testing.allocator.free(ours);
            try std.testing.expectEqualSlices(u8, ours, try t.wrapper());
        }
    }
}
