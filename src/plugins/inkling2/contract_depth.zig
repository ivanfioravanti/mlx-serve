//! Kernel contract registrations for `inkling2_depth_attn`: learned queries over each token's depth sources, at the
//! pack's width (2304) and the bank sizes and query counts its layers launch.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const depth_attn = @import("depth_attn.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const launch = metal.Launch{
    .name = "inkling2_depth_attn",
    .inputs = &depth_attn.INPUTS,
    .outputs = &depth_attn.OUTPUTS,
    .source = depth_attn.SOURCE,
    .tg_bytes = @intCast(depth_attn.tgBytes(depth_attn.MAX_BANK, depth_attn.MAX_QUERIES, 2304)),
    .m1 = 1024,
    .m2 = 1024,
};

/// A shape's rows are the token rows, n the bank's block states, k the hidden width.
const Depth = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    queries: usize,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// The bank's states then the partial one, each `[rows, k]`.
    sources: []f64 = &.{},
    query: []f64 = &.{},
    scale: f64 = 0,
    args: [depth_attn.INPUTS.len]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, queries: usize) Depth {
        return .{ .rig = metal.Rig.init(a), .launch = l, .queries = queries, .arena = .init(a) };
    }

    pub fn deinit(self: *Depth) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Depth, shape: Shape) ?metal.Geometry {
        if (shape.n > depth_attn.MAX_BANK or shape.k < 32) return null;
        const tg: c_int = @intCast(depth_attn.threads(shape.k));
        var g = metal.Geometry{ .grid = .{ ci(shape.rows) * tg, 1, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(self.queries), ci(shape.k) }, .float32);
        g.int("K", shape.n);
        g.int("D", shape.k);
        g.int("Q", self.queries);
        g.int("TG", tg);
        return g;
    }

    pub fn prepare(self: *Depth, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = shape.k;
        const per = shape.rows * d;
        const n = shape.n + 1;
        if (fill == .stress) {
            // Integer rows whose dots with the query are 0, -31 or -62 while their partial sums pass fp16's range
            // and bf16's precision, and stay under 2^24.
            self.sources = try host.ints(a, rng, n * per, 0, 127);
            self.query = try a.alloc(f64, self.queries * d);
            const half = d / 2;
            for (self.query, 0..) |*x, i| x.* = if (i % d < half) 31 else -31;
            var r: usize = 0;
            while (r < self.sources.len) : (r += d) for (0..half) |i| {
                self.sources[r + half + i] = self.sources[r + (i + 1) % half] + @as(f64, if (i < 2) @floatFromInt(rng.uintLessThan(u32, 2)) else 0);
            };
        } else {
            self.sources = try host.uniform(a, rng, n * per, -2, 2, .float32);
            self.query = try host.uniform(a, rng, self.queries * d, -4, 4, .float32);
        }
        // NaN and infinities in the first rows of the partial state and of the first block state.
        if (fill == .nonfinite) for (0..@min(n, 2)) |i| host.plant(self.sources[(n - 1 - i) * per ..][0..per], shape.rows, d);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, @floatFromInt(d))));

        const rig = &self.rig;
        rig.reset();
        const sh = [_]c_int{ 1, ci(shape.rows), ci(d) };
        const partial = try rig.input(.float32, &sh, self.sources[shape.n * per ..]);
        self.args = @splat(partial);
        for (0..shape.n) |i| self.args[i] = try rig.input(.float32, &sh, self.sources[i * per ..][0..per]);
        self.args[depth_attn.MAX_BANK + 1] = try rig.input(.float32, &.{ ci(self.queries), ci(d) }, self.query);
        self.args[depth_attn.MAX_BANK + 2] = try rig.input(.float32, &.{1}, &.{self.scale});
        return true;
    }

    pub fn run(self: *Depth) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Depth) usize {
        return self.shape.rows * self.queries * self.shape.k;
    }

    /// One softmax block over the sources, in float64. The kernel divides each weight by the sum before it mixes
    /// the sources, so each output carries the weights' rounding and one more short fp32 sum.
    fn emit(self: *const Depth, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const a = self.rig.a;
        const d = self.shape.k;
        const per = self.shape.rows * d;
        const n = self.shape.n + 1;
        const tg = depth_attn.threads(d);
        // A thread's own sum, the lanes' tree, then the simdgroups in order.
        const chain = d / tg + 5 + tg / 32 + 1;
        const terms = try a.alloc(f64, d);
        defer a.free(terms);
        const o = try a.alloc(f64, 2 * d);
        defer a.free(o);
        var keys: [depth_attn.MAX_BANK + 1]attn.Key = undefined;
        for (0..self.shape.rows) |r| for (0..self.queries) |j| {
            const q = self.query[j * d ..][0..d];
            for (keys[0..n], 0..) |*key, i| {
                const x = self.sources[i * per + r * d ..][0..d];
                const s = attn.dot(x, q, self.scale, chain, 2, sim, terms);
                key.* = .{ .s = s.v, .e = s.e, .v = x };
            }
            const st = attn.online(.{}, .{ .m = @as(f32, -3.0e38) }, keys[0..n], &.{n}, o[0..d], o[d..]);
            attn.normalize(st, o[0..d], o[d..]);
            const out = (r * self.queries + j) * d;
            @memcpy(want[out..][0..d], o[0..d]);
            if (mag) |m| for (m[out..][0..d], o[d..], 0..) |*x, e, col| {
                var mixed: f64 = 0;
                for (keys[0..n]) |key| mixed += @abs(@exp(key.s - st.m) / st.l * key.v[col]);
                x.* = host.allowance(e + @as(f64, @floatFromInt(n + 2)) * host.U * mixed);
            };
        };
    }

    pub fn reference(self: *const Depth, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Depth, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Depth) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const got = (try depth_attn.fused(&g, self.args[0..self.shape.n], self.args[depth_attn.MAX_BANK], self.args[depth_attn.MAX_BANK + 1])) orelse return error.Declined;
        return self.rig.bytesOf(&.{got});
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

fn depthSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 32, .min = 32 } },
        .served = served,
        .sums = .{ .exact = false },
    };
}

fn shapes(comptime list: []const [3]usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |x, i| out[i] = .{ .rows = x[0], .n = x[1], .k = x[2] };
    const final = out;
    return &final;
}

/// A decode row and a short chunk at the pack's width: one query (a sliding layer's inputs, the MLP's, the head's)
/// over an empty, a part-filled and a full bank; three (a global layer's q, k and v) and four (a loop message).
/// Each list opens with four or more sources: gate 4 reads the first shape, and two sources whose dots tie stay
/// tied in any accumulator.
const registrations = .{
    .{ "inkling2_depth_attn (1 query)", 1, shapes(&.{ .{ 1, 4, 2304 }, .{ 1, 0, 2304 }, .{ 9, 8, 2304 } }) },
    .{ "inkling2_depth_attn (3 queries)", 3, shapes(&.{ .{ 1, 3, 2304 }, .{ 9, 5, 2304 } }) },
    .{ "inkling2_depth_attn (4 queries)", 4, shapes(&.{ .{ 1, 5, 2304 }, .{ 1, 2, 2304 }, .{ 9, 2, 2304 } }) },
};

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(Depth, .init(set.a, &launch, r[1]), depthSpec(r[0], r[2]), null);
}

// ── tests ──

test "kernel contract inkling2 depth: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Depth = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            try t.run();
            const ours = try std.testing.allocator.dupe(u8, t.rig.outputBytes());
            defer std.testing.allocator.free(ours);
            try std.testing.expectEqualSlices(u8, ours, try t.wrapper());
        }
    }
}
