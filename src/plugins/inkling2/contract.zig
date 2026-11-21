//! Kernel contract registrations for inkling2: `inkling2_rel_attn`, the prefill attention with a relative bias and
//! a sink per query row, at the 4-bit pack's widths (3 query heads a KV head, head 128, bias table 1024, window
//! 1024 or none) and its three kinds of key pass; and its TensorOps arm `inkling2_rel_attn_nax`, on the SDK's NAX
//! contract type.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const device = @import("../../mlx/device.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const rel_attn = @import("rel_attn.zig");
const depth = @import("contract_depth.zig");
const gates = @import("contract_gate.zig");
const l2clip = @import("contract_l2clip.zig");
const conv = @import("contract_sconv.zig");
const rel_attn_nax = @import("rel_attn_nax.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const BQ = rel_attn.BQ;
const BK = rel_attn.BK;
const D = 128;
const GQ = 3;
const LOG2E = 1.44269504088896340736;

const launch = metal.Launch{
    .name = "inkling2_rel_attn",
    .inputs = &rel_attn.INPUTS,
    .outputs = &rel_attn.OUTPUTS,
    .source = rel_attn.SOURCE,
    .header = rel_attn.HEADER,
    .contiguous = false,
    .tg_bytes = @intCast(rel_attn.tgBytes(D)),
    .m1 = 512,
    .m2 = 512,
};

/// One dispatch of the key range: all of it, or a pass that hands its state on (out), takes and hands it
/// (through), or takes it and normalizes (in); or every key chunk at once, each from a fresh state (split).
const Phase = enum { single, out, through, in, split };

const Cfg = struct {
    dt: mlx.mlx_dtype = .bfloat16,
    rel_dt: mlx.mlx_dtype = .bfloat16,
    r: usize = 1024,
    window: usize = 0,
    /// The row's first real key, in thirds of the keys.
    start: usize = 0,
    phase: Phase = .single,
};

/// A shape's rows are the KV heads (`GQ` query heads each), n the chunk's rows, k the keys. K and V are the cache's
/// views: the first k rows of a longer buffer.
const RelAttn = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Cfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    rel: []f64 = &.{},
    sinks: []f64 = &.{},
    m_in: []f64 = &.{},
    l_in: []f64 = &.{},
    o_in: []f64 = &.{},
    scale: f64 = 0,
    start: usize = 0,
    kr: [2]usize = .{ 0, 0 },
    args: [rel_attn.INPUTS.len]mlx.mlx_array = undefined,
    views: [2]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, l: *const metal.Launch, cfg: Cfg) RelAttn {
        return .{ .rig = metal.Rig.init(a), .launch = l, .cfg = cfg, .arena = .init(a) };
    }

    fn freeViews(self: *RelAttn) void {
        for (&self.views) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *RelAttn) void {
        self.freeViews();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn final(phase: Phase) bool {
        return phase == .single or phase == .in;
    }

    pub fn geometry(self: *const RelAttn, shape: Shape) ?metal.Geometry {
        return self.geometryAt(shape, self.cfg.phase);
    }

    fn geometryAt(self: *const RelAttn, shape: Shape, phase: Phase) ?metal.Geometry {
        const c = self.cfg;
        const least: usize = if (phase == .split) 1 else rel_attn.MIN_ROWS;
        if (shape.n < least or shape.k < shape.n) return null;
        const hq: c_int = @intCast(shape.rows * GQ);
        const l: c_int = @intCast(shape.n);
        const nch: c_int = if (phase == .split) @intCast(chunks(shape.k)) else 1;
        var g = metal.Geometry{ .grid = .{ @divTrunc(l + BQ - 1, BQ) * nch * rel_attn.TG, hq, 1 }, .tg = .{ rel_attn.TG, 1, 1 } };
        if (phase == .split) {
            g.out(&.{1}, c.dt);
            g.out(&.{ nch, l * hq }, .float32);
            g.out(&.{ nch, l * hq }, .float32);
            g.out(&.{ nch, l * hq, D }, .float32);
            g.unread = 0b0001;
        } else if (final(phase)) {
            g.out(&.{ 1, l, hq, D }, c.dt);
            for (0..3) |_| g.out(&.{1}, .float32);
            g.unread = 0b1110;
        } else {
            g.out(&.{1}, c.dt);
            g.out(&.{ 1, l, hq }, .float32);
            g.out(&.{ 1, l, hq }, .float32);
            g.out(&.{ 1, l, hq, D }, .float32);
            g.unread = 0b0001;
        }
        g.dtype("T", c.dt);
        g.int("D", D);
        g.int("HQ", hq);
        g.int("HKV", shape.rows);
        g.int("R", c.r);
        g.int("WINDOW", c.window);
        g.int("BQ", BQ);
        g.int("TG", rel_attn.TG);
        return g;
    }

    /// Key chunks a split launch makes of `s` keys.
    fn chunks(s: usize) usize {
        const each: usize = @intCast(rel_attn.chunkKeys(@intCast(s)));
        return (s + each - 1) / each;
    }

    /// The pass length the chain splits the keys at: whole blocks, a third of the keys.
    fn chunk(s: usize) usize {
        return @max(BK, s / 3 / BK * BK);
    }

    fn rangeOf(phase: Phase, s: usize) [2]usize {
        const c = chunk(s);
        return switch (phase) {
            .single => .{ 0, s },
            .out => .{ 0, c },
            .through => .{ c, 2 * c },
            .in => .{ 2 * c, s },
            // The kernel reads a chunk's keys where a pass reads its first key.
            .split => .{ @intCast(rel_attn.chunkKeys(@intCast(s))), 0 },
        };
    }

    pub fn prepare(self: *RelAttn, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeViews();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const hk = shape.rows;
        const hq = hk * GQ;
        const l = shape.n;
        const s = shape.k;
        if (fill == .stress) {
            self.q = try a.alloc(f64, hq * l * D);
            self.k = try host.ints(a, rng, hk * s * D, 0, 127);
            stressRows(rng, self.q, self.k);
        } else {
            self.q = try host.uniform(a, rng, hq * l * D, -2, 2, c.dt);
            self.k = try host.uniform(a, rng, hk * s * D, -2, 2, c.dt);
        }
        self.v = try host.uniform(a, rng, hk * s * D, -2, 2, c.dt);
        if (fill == .nonfinite) {
            // A query row's own NaN / Inf (rows 0..3 of head 0), and the value rows of the first keys.
            host.plant(self.q, hq * l, D);
            host.plant(self.v, hk * s, D);
        }
        self.rel = try host.uniform(a, rng, l * hq * c.r, -3, 3, c.rel_dt);
        self.sinks = try host.uniform(a, rng, l * hq, -1, 3, .float32);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, D)));
        self.start = c.start * s / 3;
        self.kr = rangeOf(c.phase, s);
        self.m_in = try host.uniform(a, rng, l * hq, -2, 3, .float32);
        self.l_in = try host.uniform(a, rng, l * hq, 1, 8, .float32);
        self.o_in = try host.uniform(a, rng, l * hq * D, -8, 8, .float32);
        try self.bind();
        return true;
    }

    fn sizesOf(rig: *metal.Rig, l: usize, s: usize, kr: [2]usize, phase: Phase) !mlx.mlx_array {
        const bits = @as(u32, @intFromBool(phase == .through or phase == .in)) | (@as(u32, @intFromBool(final(phase))) << 1) | (@as(u32, @intFromBool(phase == .split)) << 2);
        return rig.input(.int32, &.{5}, &.{ @floatFromInt(l), @floatFromInt(s), @floatFromInt(kr[0]), @floatFromInt(kr[1]), @floatFromInt(bits) });
    }

    fn bind(self: *RelAttn) !void {
        const rig = &self.rig;
        rig.reset();
        const c = self.cfg;
        const hk = self.shape.rows;
        const hq = hk * GQ;
        const l = self.shape.n;
        const s = self.shape.k;
        const a = self.arena.allocator();
        const q = try rig.input(c.dt, &.{ 1, ci(hq), ci(l), D }, self.q);
        const cap = s + 24;
        var pad = std.Random.DefaultPrng.init(91);
        for ([_][]const f64{ self.k, self.v }, 0..) |src, i| {
            const buf = try host.uniform(a, pad.random(), hk * cap * D, -1, 1, c.dt);
            for (0..hk) |h| @memcpy(buf[h * cap * D ..][0 .. s * D], src[h * s * D ..][0 .. s * D]);
            const full = try rig.input(c.dt, &.{ 1, ci(hk), ci(cap), D }, buf);
            self.views[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&self.views[i], full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ci(hk), ci(s), D }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, rig.stream()));
        }
        const dummy = try rig.input(.float32, &.{1}, &.{0});
        const carry = c.phase == .through or c.phase == .in;
        self.args = .{
            q,
            self.views[0],
            self.views[1],
            try rig.input(c.rel_dt, &.{ 1, ci(l), ci(hq), ci(c.r) }, self.rel),
            try rig.input(.float32, &.{ 1, ci(l), ci(hq) }, self.sinks),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(self.start)}),
            try sizesOf(rig, l, s, self.kr, c.phase),
            try rig.input(.float32, &.{1}, &.{self.scale}),
            if (carry) try rig.input(.float32, &.{ 1, ci(l), ci(hq) }, self.m_in) else dummy,
            if (carry) try rig.input(.float32, &.{ 1, ci(l), ci(hq) }, self.l_in) else dummy,
            if (carry) try rig.input(.float32, &.{ 1, ci(l), ci(hq), D }, self.o_in) else dummy,
        };
    }

    pub fn run(self: *RelAttn) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RelAttn) usize {
        const rows = self.shape.rows * GQ * self.shape.n;
        if (self.cfg.phase == .split) return chunks(self.shape.k) * rows * (2 + D);
        return if (final(self.cfg.phase)) rows * D else rows * (2 + D);
    }

    /// Each row's blocks as the kernel walks them, in float64: a hidden key scores -Inf, a visible one q.k plus
    /// its bias, whose product and sum are two more fp32 roundings.
    fn emit(self: *const RelAttn, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const a = self.rig.a;
        const c = self.cfg;
        const hq = self.shape.rows * GQ;
        const l = self.shape.n;
        const s = self.shape.k;
        const blocks = (s + BK - 1) / BK;
        const keys = try a.alloc(attn.Key, blocks * BK);
        defer a.free(keys);
        const ends = try a.alloc(usize, blocks);
        defer a.free(ends);
        const zeros: [D]f64 = @splat(0);
        var terms: [D]f64 = undefined;
        var o: [D]f64 = undefined;
        var eo: [D]f64 = undefined;
        const scale2 = self.scale * LOG2E;
        const carry = c.phase == .through or c.phase == .in;
        const nch = if (c.phase == .split) chunks(s) else 1;
        for (0..nch) |ch| for (0..l) |r| for (0..hq) |h| {
            const kr: [2]usize = if (c.phase == .split) .{ ch * self.kr[0], (ch + 1) * self.kr[0] } else self.kr;
            const hk = h / GQ;
            const row = r * hq + h;
            const q_lo = s - l + r / BQ * BQ;
            const pos = s - l + r;
            const kb_lim = @min((@min(kr[1], s) + BK - 1) / BK, (q_lo + BQ - 1 + BK) / BK);
            var kb = @max(kr[0], self.start) / BK;
            if (c.window > 0) kb = @max(kb, (q_lo + 1 -| c.window) / BK);
            const qrow = self.q[(h * l + r) * D ..][0..D];
            var n: usize = 0;
            var nb: usize = 0;
            while (kb < kb_lim) : (kb += 1) {
                for (0..BK) |j| {
                    const key = kb * BK + j;
                    const v = if (key < s) self.v[(hk * s + key) * D ..][0..D] else &zeros;
                    const hide = key >= s or key > pos or key < self.start or (c.window > 0 and pos - key >= c.window);
                    keys[n] = if (hide) .{ .s = -std.math.inf(f64), .on = false, .v = v } else blk: {
                        const d = attn.dot(qrow, self.k[(hk * s + key) * D ..][0..D], scale2, D, 3, sim, &terms);
                        if (pos - key >= c.r) break :blk .{ .s = d.v, .e = d.e, .v = v };
                        const b = self.rel[row * c.r + pos - key] * LOG2E;
                        break :blk .{ .s = d.v + b, .e = d.e + host.U * (3 * @abs(b) + @abs(d.v + b)), .v = v };
                    };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const sink = self.sinks[row] * LOG2E;
            const first: attn.Init = if (carry)
                .{ .m = self.m_in[row], .l = self.l_in[row], .o = self.o_in[row * D ..][0..D] }
            else if (ch > 0)
                // Only the first chunk holds the sink.
                .{ .m = @as(f32, -3.0e38) }
            else
                .{ .m = sink, .e = 2 * host.U * @abs(sink), .l = 1 };
            const st = attn.online(.{ .base2 = true }, first, keys[0..n], ends[0..nb], &o, &eo);
            if (final(c.phase)) {
                attn.normalize(st, &o, &eo);
                @memcpy(want[row * D ..][0..D], &o);
                if (mag) |m| for (m[row * D ..][0..D], eo) |*x, e| {
                    x.* = host.allowance(e);
                };
                continue;
            }
            const rows = nch * hq * l;
            const at = ch * hq * l + row;
            want[at] = st.m;
            want[rows + at] = st.l;
            @memcpy(want[2 * rows + at * D ..][0..D], &o);
            if (mag) |m| {
                m[at] = host.allowance(st.em);
                m[rows + at] = host.allowance(st.el);
                for (m[2 * rows + at * D ..][0..D], eo) |*x, e| x.* = host.allowance(e);
            }
        };
    }

    pub fn reference(self: *const RelAttn, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const RelAttn, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs, its key passes cut at `budget` row-key pairs (0: one dispatch).
    fn wrapper(self: *RelAttn, budget: i64) ![]const u8 {
        rel_attn.budget_override = budget;
        defer rel_attn.budget_override = null;
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const got = (try rel_attn.simd(&g, .{ .q = x[0], .k = x[1], .v = x[2], .rel = x[3], .sinks = x[4], .key_start = x[5], .scale = @floatCast(self.scale), .window = @intCast(self.cfg.window) })) orelse return error.Declined;
        return self.rig.bytesOf(&.{got});
    }

    /// The shipped key-split wrapper on the same inputs.
    fn splitWrapper(self: *RelAttn) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const got = (try rel_attn.split(&g, .{ .q = x[0], .k = x[1], .v = x[2], .rel = x[3], .sinks = x[4], .key_start = x[5], .scale = @floatCast(self.scale), .window = @intCast(self.cfg.window) })) orelse return error.Declined;
        return self.rig.bytesOf(&.{got});
    }

    /// The declared launch run as the chain of passes: out, through, in, each pass's state read back and bound as
    /// the next one's carry. The last output's bytes.
    fn chained(self: *RelAttn, a: std.mem.Allocator) ![]u8 {
        const rig = &self.rig;
        const hq = ci(self.shape.rows * GQ);
        const l = ci(self.shape.n);
        var args = self.args;
        for ([_]Phase{ .out, .through, .in }) |phase| {
            args[6] = try sizesOf(rig, self.shape.n, self.shape.k, rangeOf(phase, self.shape.k), phase);
            try rig.launch(self.launch, self.geometryAt(self.shape, phase).?, &args);
            if (phase == .in) break;
            const m = try host.read(a, rig.output(1), rig.stream());
            defer a.free(m);
            const sum = try host.read(a, rig.output(2), rig.stream());
            defer a.free(sum);
            const o = try host.read(a, rig.output(3), rig.stream());
            defer a.free(o);
            args[8] = try rig.input(.float32, &.{ 1, l, hq }, m);
            args[9] = try rig.input(.float32, &.{ 1, l, hq }, sum);
            args[10] = try rig.input(.float32, &.{ 1, l, hq, D }, o);
        }
        return a.dupe(u8, rig.outputBytes());
    }
};

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// Gate 4's rows, as `sdk/blocks/attention/contract.zig` builds them: q is +251 on its first half and -251 on its
/// second; each drawn k row's second half is its first rotated by one, plus 1 at up to two places. A dot is 0,
/// -251 or -502 while its partial sums pass fp16's range and bf16's precision.
fn stressRows(rng: std.Random, q: []f64, k: []f64) void {
    const half = D / 2;
    for (q, 0..) |*x, i| x.* = if (i % D < half) 251 else -251;
    var r: usize = 0;
    while (r < k.len) : (r += D) for (0..half) |i| {
        k[r + half + i] = k[r + (i + 1) % half] + @as(f64, if (i < 2) @floatFromInt(rng.uintLessThan(u32, 2)) else 0);
    };
}

fn relSpec(comptime name: []const u8, cfg: Cfg, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = if (RelAttn.final(cfg.phase)) host.elemOf(cfg.dt) else .f32,
        .tiles = .{ .{}, .{ .size = BQ, .min = if (cfg.phase == .split) 1 else rel_attn.MIN_ROWS }, .{ .size = BK, .min = rel_attn.MIN_ROWS } },
        .served = served,
        .sums = .{ .exact = false },
    };
}

/// rows = KV heads, n = chunk rows, k = keys.
fn qk(comptime list: []const [3]usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |x, i| out[i] = .{ .rows = x[0], .n = x[1], .k = x[2] };
    const final = out;
    return &final;
}

/// The pack's sliding layers (window 1024 = the bias table) and global ones (no window, keys past the table), a
/// ring that is not yet full (the first key a third in), the chain's three passes, an f16 pack and an fp32 bias.
/// The chunks are short: the float64 reference walks every row-key pair.
const registrations = .{
    .{ "inkling2_rel_attn bf16 (window 1024)", Cfg{ .window = 1024 }, qk(&.{ .{ 1, 16, 1100 }, .{ 1, 65, 1090 } }) },
    .{ "inkling2_rel_attn bf16 (global)", Cfg{}, qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 }, .{ 2, 130, 200 } }) },
    .{ "inkling2_rel_attn bf16 (global, keys past the bias table)", Cfg{}, qk(&.{.{ 1, 16, 1100 }}) },
    .{ "inkling2_rel_attn bf16 (global, first key a third in)", Cfg{ .start = 1 }, qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 } }) },
    .{ "inkling2_rel_attn bf16 (global, pass hands its state on)", Cfg{ .phase = .out }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }) },
    .{ "inkling2_rel_attn bf16 (global, pass carries the state through)", Cfg{ .phase = .through }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }) },
    .{ "inkling2_rel_attn bf16 (global, last pass takes the state)", Cfg{ .phase = .in }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }) },
    .{ "inkling2_rel_attn bf16 (global, key chunks from fresh states)", Cfg{ .phase = .split }, qk(&.{ .{ 1, 6, 2300 }, .{ 1, 1, 200 }, .{ 1, 16, 1100 } }) },
    .{ "inkling2_rel_attn bf16 (window 1024, key chunks from fresh states)", Cfg{ .phase = .split, .window = 1024 }, qk(&.{.{ 1, 9, 2100 }}) },
    .{ "inkling2_rel_attn f16 (global)", Cfg{ .dt = .float16, .rel_dt = .float16 }, qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 } }) },
    .{ "inkling2_rel_attn bf16 (global, fp32 bias)", Cfg{ .rel_dt = .float32 }, qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 } }) },
};

/// The TensorOps arm: the SDK's NAX attention with its relative-bias entry, one dispatch (WN 1) or key passes (WN 2).
fn naxLaunch(comptime wn: u32) metal.Launch {
    return .{
        .name = "inkling2_rel_attn_nax",
        .inputs = &rel_attn_nax.INPUTS,
        .outputs = &rel_attn_nax.OUTPUTS,
        .source = rel_attn_nax.SOURCE,
        .header = rel_attn_nax.HEADER,
        .contiguous = false,
        .tg_bytes = if (wn == 2) 2 * 2 * 16 * 32 * 4 else 4,
        .offline = false,
        .min_gen = 17,
    };
}

const nax_one = naxLaunch(1);
const nax_two = naxLaunch(2);

fn naxCfg(window: usize, start: usize, phase: attn.Phase) attn.NaxCfg {
    return .{ .dk = D, .gqa = GQ, .rel = 1024, .window = window, .start = start, .phase = phase };
}

const nax_registrations = .{
    .{ "inkling2_rel_attn_nax bf16 (window 1024)", naxCfg(1024, 0, .single), qk(&.{ .{ 1, 16, 1100 }, .{ 1, 65, 1090 } }), &nax_one },
    .{ "inkling2_rel_attn_nax bf16 (global)", naxCfg(0, 0, .single), qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 }, .{ 2, 130, 200 } }), &nax_one },
    .{ "inkling2_rel_attn_nax bf16 (global, keys past the bias table)", naxCfg(0, 0, .single), qk(&.{.{ 1, 16, 1100 }}), &nax_one },
    .{ "inkling2_rel_attn_nax bf16 (global, first key a third in)", naxCfg(0, 1, .single), qk(&.{ .{ 1, 16, 200 }, .{ 1, 80, 300 } }), &nax_one },
    .{ "inkling2_rel_attn_nax bf16 (global, first key-range pass)", naxCfg(0, 0, .out), qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
    .{ "inkling2_rel_attn_nax bf16 (global, middle key-range pass)", naxCfg(0, 0, .through), qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
    .{ "inkling2_rel_attn_nax bf16 (global, last key-range pass)", naxCfg(0, 0, .in), qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
};


// ── inkling2_rel_attn_merge: a split launch's chunk states joined ──

const merge_launch = metal.Launch{
    .name = "inkling2_rel_attn_merge",
    .inputs = &rel_attn.MERGE_INPUTS,
    .outputs = &.{"out"},
    .source = rel_attn.MERGE_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// A shape's rows are the query rows of every head, n the key chunks, k the head width.
const Merge = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    ms: []f64 = &.{},
    ls: []f64 = &.{},
    os: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, l: *const metal.Launch, dt: mlx.mlx_dtype) Merge {
        return .{ .rig = metal.Rig.init(a), .launch = l, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Merge) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Merge, shape: Shape) ?metal.Geometry {
        if (shape.n < 1 or shape.n > rel_attn.MAX_CHUNKS) return null;
        var g = metal.Geometry{ .grid = .{ ci(shape.rows * 32), 1, 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ ci(shape.rows), ci(shape.k) }, self.dt);
        g.dtype("T", self.dt);
        g.int("D", shape.k);
        g.int("CH", shape.n);
        return g;
    }

    pub fn prepare(self: *Merge, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const cells = shape.n * shape.rows;
        self.ms = try host.uniform(a, rng, cells, -3, 3, .float32);
        self.ls = try host.uniform(a, rng, cells, 0.5, 4, .float32);
        self.os = try host.uniform(a, rng, cells * shape.k, -4, 4, .float32);
        // A chunk past the first that saw no key: the kernel's empty state.
        for (shape.rows..cells) |i| if (rng.uintLessThan(u32, 3) == 0) {
            self.ms[i] = @as(f32, -3.0e38);
            self.ls[i] = 0;
            @memset(self.os[i * shape.k ..][0..shape.k], 0);
        };
        if (fill == .nonfinite) host.plant(self.os[0 .. shape.rows * shape.k], shape.rows, shape.k);
        try self.bind();
        return true;
    }

    fn bind(self: *Merge) !void {
        const rig = &self.rig;
        const shape = self.shape;
        rig.reset();
        self.args = .{
            try rig.input(.float32, &.{ ci(shape.n), ci(shape.rows) }, self.ms),
            try rig.input(.float32, &.{ ci(shape.n), ci(shape.rows) }, self.ls),
            try rig.input(.float32, &.{ ci(shape.n), ci(shape.rows), ci(shape.k) }, self.os),
        };
    }

    pub fn run(self: *Merge) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Merge) usize {
        return self.shape.rows * self.shape.k;
    }

    fn emit(self: *const Merge, want: []f64, mag: ?[]f64, sim: attn.Sim) void {
        const rows = self.shape.rows;
        const d = self.shape.k;
        const nch = self.shape.n;
        var w: [rel_attn.MAX_CHUNKS]f64 = undefined;
        var ew: [rel_attn.MAX_CHUNKS]f64 = undefined;
        var terms: [rel_attn.MAX_CHUNKS]f64 = undefined;
        const sums = spec.gamma(@floatFromInt(nch + 1), host.U);
        for (0..rows) |row| {
            var top = -std.math.inf(f64);
            for (0..nch) |c| top = @max(top, self.ms[c * rows + row]);
            var eden: f64 = 0;
            for (0..nch) |c| {
                const span = top - self.ms[c * rows + row];
                w[c] = @exp2(-span);
                // A fast exp2 and the product.
                ew[c] = host.fastExp(span * std.math.ln2) + 2 * host.U + sums;
                terms[c] = self.ls[c * rows + row] * w[c];
                eden += @abs(terms[c]) * ew[c];
            }
            const den = if (sim) |x| spec.sumIn(x.acc, x.order, terms[0..nch]) else host.sum(terms[0..nch]);
            for (0..d) |col| {
                var enum_: f64 = 0;
                for (0..nch) |c| {
                    terms[c] = self.os[(c * rows + row) * d + col] * w[c];
                    enum_ += @abs(terms[c]) * ew[c];
                }
                const num = if (sim) |x| spec.sumIn(x.acc, x.order, terms[0..nch]) else host.sum(terms[0..nch]);
                const out = num / den;
                want[row * d + col] = out;
                if (mag) |m| m[row * d + col] = host.allowance((enum_ + @abs(out) * eden) / @abs(den) + 2 * host.U * @abs(out));
            }
        }
    }

    pub fn reference(self: *const Merge, want: []f64, mag: []f64) !void {
        self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Merge, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

fn mergeSpec(comptime name: []const u8, dt: mlx.mlx_dtype, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = host.elemOf(dt),
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a row sums a few dozen chunk states of fp32 weights: the output's own rounding is coarser", .exact = false },
    };
}

/// A decode-width window over three chunks, a full window's rows over the most chunks, one chunk.
const merge_shapes = qk(&.{ .{ 18, 3, 128 }, .{ 48, 64, 128 }, .{ 3, 1, 128 } });

pub fn register(set: *kc.Set) !void {
    inline for (registrations) |r| try set.add(RelAttn, .init(set.a, &launch, r[1]), relSpec(r[0], r[1], r[2]), null);
    inline for (nax_registrations) |r| try set.add(attn.NaxAttn, .init(set.a, r[3], r[1]), attn.naxSpec(r[0], r[1], r[2]), null);
    inline for (.{ .{ "bf16", mlx.mlx_dtype.bfloat16 }, .{ "f16", mlx.mlx_dtype.float16 } }) |t| try set.add(Merge, .init(set.a, &merge_launch, t[1]), mergeSpec("inkling2_rel_attn_merge " ++ t[0], t[1], merge_shapes), null);
    try depth.register(set);
    try gates.register(set);
    try l2clip.register(set);
    try conv.register(set);
}

// ── tests ──

test "kernel contract inkling2: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items[0..registrations.len]) |k| {
        const t: *RelAttn = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            const ours = switch (t.cfg.phase) {
                // The chain's first two passes ride its last one's registration.
                .out, .through => continue,
                .in => try t.chained(a),
                .split => blk: {
                    // The split launch's states through the declared merge.
                    try t.run();
                    var m = Merge.init(a, &merge_launch, t.cfg.dt);
                    defer m.deinit();
                    const rows = shape.rows * GQ * shape.n;
                    m.shape = .{ .rows = rows, .n = RelAttn.chunks(shape.k), .k = D };
                    m.ms = try host.read(m.arena.allocator(), t.rig.output(1), t.rig.stream());
                    m.ls = try host.read(m.arena.allocator(), t.rig.output(2), t.rig.stream());
                    m.os = try host.read(m.arena.allocator(), t.rig.output(3), t.rig.stream());
                    try m.bind();
                    try m.run();
                    break :blk try a.dupe(u8, m.rig.outputBytes());
                },
                .single => blk: {
                    try t.run();
                    break :blk try a.dupe(u8, t.rig.outputBytes());
                },
            };
            defer a.free(ours);
            const budget: i64 = if (t.cfg.phase == .in) @intCast(shape.rows * GQ * shape.n * RelAttn.chunk(shape.k)) else 0;
            try std.testing.expectEqualSlices(u8, ours, if (t.cfg.phase == .split) try t.splitWrapper() else try t.wrapper(budget));
        }
    }
}

test "kernel contract inkling2: the NAX launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items[registrations.len..][0..nax_registrations.len]) |k| {
        const t: *attn.NaxAttn = @ptrCast(@alignCast(k.ctx));
        // The chain's first two passes ride its last one's registration.
        if (t.cfg.phase == .out or t.cfg.phase == .through) continue;
        for (k.spec.served) |shape| {
            try std.testing.expect(try t.prepare(shape, .random, 7));
            const ours = try t.declared(a);
            defer a.free(ours);
            const nk: c_int = @intCast((shape.k + 31) / 32);
            rel_attn_nax.edges_override = if (t.cfg.phase == .in) &.{ 0, @divTrunc(nk, 3), @divTrunc(2 * nk, 3), nk } else &.{ 0, nk };
            defer rel_attn_nax.edges_override = null;
            var g = ops.Graph.init(a, t.rig.stream());
            defer g.deinit();
            const x = t.args;
            const got = (try rel_attn_nax.nax(&g, .{ .q = x[0], .k = x[1], .v = x[2], .rel = x[6], .sinks = x[3], .key_start = x[7], .scale = @floatCast(t.scale), .window = @intCast(t.cfg.window) })) orelse return error.Declined;
            try std.testing.expectEqualSlices(u8, ours, try t.rig.bytesOf(&.{try g.transpose(got, &.{ 0, 2, 1, 3 })}));
        }
    }
}
