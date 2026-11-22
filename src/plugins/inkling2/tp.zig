//! Inkling 2's experts across a tensor-parallel pair (`--dist tensor`): both ranks hold every bank whole, and rank 0
//! splits each long prefill chunk's (row, expert) pairs by expert, from the routing it has: the worker takes about
//! `WORKER_SHARE` of the pairs, rank 0 the rest, each through full-width gathers. The worker sends back its rows'
//! weighted sums, which rank 0 adds to its own in f32.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const host_abi = @import("../../abi/host.zig");
const layers = @import("layers.zig");
const kernels = @import("kernels.zig");
const Config = @import("config.zig").Config;

const Graph = ops.Graph;
const Array = ops.Array;
const gpa = std.heap.c_allocator;

/// Prefill chunks from this many rows split their experts with the worker: past where the split overtakes one node.
/// Decode, verify rounds and short spans never cross the wire.
pub const TP_ROWS = 2048;

/// The worker's part of a chunk's pairs: its pairs plus the wire both ways take what rank 0's pairs take.
/// ponytail: one fraction for this pair of Macs and link; learn it from the timing lines if another pair differs.
pub const WORKER_SHARE: f32 = 0.42;

/// Rank 0's link to the worker: the host whose `expert_exchange` reaches it, and whether it still answers (a failed
/// exchange leaves the rest on this node; the server restarts to pair again).
pub const Pair = struct {
    host: *const host_abi.Host,
    live: bool = true,
    /// Per `TIMING_LINE` exchanges, rank 0's host time around each (µs): the routing sync, the split and the sum
    /// kernel's places, the wait for the worker.
    timing: struct { n: u32 = 0, sync: i64 = 0, host: i64 = 0, wait: i64 = 0 } = .{},
};

const TIMING_LINE = 48;

fn nowUs() i64 {
    return @import("../../sdk/util/io_util.zig").nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

/// Rank 0's tie from a layer to the pair.
pub const Tp = struct { pair: *Pair, layer: u32 };

/// A rank's pairs, sorted by expert: each one's place `row * k + pick` in the routing, its expert and its weight.
pub const Pairs = struct { at: []u32, sel: []u32, w: []f32 };

/// The pairs of `idx` / `weights` (`[t * k]`, row-major) split by expert: `worker` gets the experts it is given
/// greedily, most pairs first, while its pairs stay within `share` of all; `mine` the rest. Both sorted by expert.
pub fn split(a: std.mem.Allocator, idx: []const u32, weights: []const f32, experts: usize, share: f32) ![2]Pairs {
    const counts = try a.alloc(u32, experts);
    defer a.free(counts);
    @memset(counts, 0);
    for (idx) |e| counts[e] += 1;
    const order = try a.alloc(u32, experts);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, counts, struct {
        fn more(c: []const u32, x: u32, y: u32) bool {
            return c[x] > c[y] or (c[x] == c[y] and x < y);
        }
    }.more);
    const theirs = try a.alloc(bool, experts);
    defer a.free(theirs);
    @memset(theirs, false);
    const target: u64 = @intFromFloat(@as(f32, @floatFromInt(idx.len)) * share);
    var taken: u64 = 0;
    for (order) |e| if (counts[e] > 0 and taken + counts[e] <= target) {
        theirs[e] = true;
        taken += counts[e];
    };
    // Counting sort by expert: each expert's pairs land at its offset in its rank's list.
    const at = try a.alloc(u32, experts);
    defer a.free(at);
    var sizes = [2]u32{ 0, 0 };
    for (0..experts) |e| {
        const r = @intFromBool(theirs[e]);
        at[e] = sizes[r];
        sizes[r] += counts[e];
    }
    var out: [2]Pairs = undefined;
    for (&out, sizes) |*p, n| p.* = .{ .at = try a.alloc(u32, n), .sel = try a.alloc(u32, n), .w = try a.alloc(f32, n) };
    for (idx, weights, 0..) |e, wt, j| {
        const p = &out[@intFromBool(theirs[e])];
        const i = at[e];
        at[e] += 1;
        p.at[i] = @intCast(j);
        p.sel[i] = e;
        p.w[i] = wt;
    }
    return out;
}

pub fn free(a: std.mem.Allocator, p: Pairs) void {
    a.free(p.at);
    a.free(p.sel);
    a.free(p.w);
}

/// The down projections `[p, 1, D]` of the rows `x` `[t, D]` through the pairs `at` (`row * k + pick`) and `sel`
/// (sorted by expert), as the gather path of `kernels.experts` runs a pair.
pub fn downs(g: *Graph, L: *const layers.Layer, cfg: *const Config, x: Array, at: Array, sel: Array) !Array {
    const d = ops.shape(x)[1];
    const p = ops.shape(sel)[0];
    const caps = try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const scaled = L.gate_up_scale.ctx != null;
    const rows = try g.call(mlx.mlx_floor_divide, .{ at, try g.astype(try g.scalar(@floatFromInt(cfg.top_k), .float32), .uint32) });
    const picked = try g.reshape(try g.take(x, rows, 0), &.{ p, 1, d });
    const act = try kernels.activation(g, try layers.gather(g, picked, &L.gate_up, sel, true), sel, if (scaled) L.gate_up_scale else caps, caps, scaled);
    return layers.gather(g, act, &L.down, sel, true);
}

/// The sum kernel's view of a rank's pairs among a chunk's `n` = `t * k` places: each place's pair in the list, or
/// `zero` (a zero row) where the pair is the other rank's, and its weight times the expert's down scale (`ds`, so
/// the kernel reads no scale), 0 there.
const Places = struct { inv: []u32, w: []f32 };

fn places(a: std.mem.Allocator, p: Pairs, n: usize, zero: u32, ds: ?[]const f32) !Places {
    const inv = try a.alloc(u32, n);
    const w = try a.alloc(f32, n);
    @memset(inv, zero);
    @memset(w, 0);
    for (p.at, p.sel, p.w, 0..) |j, e, wt, i| {
        inv[j] = @intCast(i);
        w[j] = if (ds) |s| wt * s[e] else wt;
    }
    return .{ .inv = inv, .w = w };
}

/// `L.down_scale` on the host (evaluated at load), or null.
fn downScale(L: *const layers.Layer) ?[]const f32 {
    if (L.down_scale.ctx == null) return null;
    const n: usize = @intCast(ops.shape(L.down_scale)[0]);
    return (mlx.mlx_array_data_float32(L.down_scale) orelse return null)[0..n];
}

/// What `kernels.experts` hands the sum kernel for a split chunk: rank 0's pair outputs, the worker's row sums and a
/// zero row (`dn`); each row's `k` places plus one for the worker's sum (`inverse` `[t * (k + 1)]`, `weights`
/// `[1, t, k + 1]`, the down scales folded in); `picks` = k + 1.
pub const Split = struct { dn: Array, sel: Array, inverse: Array, weights: Array, picks: c_int };

/// Rank 0, a prefill chunk's experts split with the worker; null = the single-node MLP (a short chunk, no pair, the
/// worker given nothing, or gone).
pub fn exchange(g: *Graph, L: *const layers.Layer, cfg: *const Config, x: Array, idx: Array, weights: Array) !?Split {
    const tp = L.tp orelse return null;
    const pair = tp.pair;
    const s = ops.shape(x)[1];
    if (!pair.live or s < TP_ROWS) return null;
    const h = pair.host;
    const send = host_abi.expertExchangeOf(h) orelse return null;
    const d = ops.shape(x)[2];
    const k: usize = cfg.top_k;
    const t: usize = @intCast(s);
    const n = t * k;
    const flat = try g.reshape(x, &.{ s, d });
    const routes = try g.contiguous(try g.astype(try g.reshape(idx, &.{-1}), .uint32));
    const w32 = try g.contiguous(try g.astype(try g.reshape(weights, &.{-1}), .float32));
    // The split reads the routing here; the rows are ready to go once it returns.
    const t0 = nowUs();
    try evalAll(&.{ flat, routes, w32 });
    const t1 = nowUs();
    const parts = try split(gpa, (mlx.mlx_array_data_uint32(routes) orelse return error.Inkling2TpRoutes)[0..n], (mlx.mlx_array_data_float32(w32) orelse return error.Inkling2TpRoutes)[0..n], cfg.num_experts, WORKER_SHARE);
    defer for (parts) |p| free(gpa, p);
    if (parts[1].sel.len == 0) return null;
    const p0 = parts[0].sel.len;
    const p1: c_int = @intCast(parts[1].sel.len);
    const mine = if (p0 > 0) try downs(g, L, cfg, flat, try g.fromSlice(u32, parts[0].at, &.{@intCast(p0)}), try g.fromSlice(u32, parts[0].sel, &.{@intCast(p0)})) else null;
    // The worker's pairs as `[p, 2]` (place, expert) and their weights `[p]`.
    const theirs_pairs = try g.reshape(try g.concat(&.{ try g.fromSlice(u32, parts[1].at, &.{ p1, 1 }), try g.fromSlice(u32, parts[1].sel, &.{ p1, 1 }) }, 1), &.{ p1, 2 });
    const theirs_w = try g.fromSlice(f32, parts[1].w, &.{p1});
    try evalAll(&.{ theirs_pairs, theirs_w });
    if (mine) |y| try queue(&.{y});
    // Rows: rank 0's p0 outputs, the worker's t sums, one zero row; each row's last place is the worker's sum.
    const pl = try places(gpa, parts[0], n, @intCast(p0 + t), downScale(L));
    defer gpa.free(pl.inv);
    defer gpa.free(pl.w);
    const inv = try gpa.alloc(u32, t * (k + 1));
    defer gpa.free(inv);
    const wts = try gpa.alloc(f32, t * (k + 1));
    defer gpa.free(wts);
    for (0..t) |r| {
        @memcpy(inv[r * (k + 1) ..][0..k], pl.inv[r * k ..][0..k]);
        @memcpy(wts[r * (k + 1) ..][0..k], pl.w[r * k ..][0..k]);
        inv[r * (k + 1) + k] = @intCast(p0 + r);
        wts[r * (k + 1) + k] = 1;
    }
    const t2 = nowUs();
    var theirs: host_abi.Array = .{};
    const v = send(h.ctx, tp.layer, .{ .ctx = flat.ctx }, .{ .ctx = theirs_pairs.ctx }, .{ .ctx = theirs_w.ctx }, &theirs);
    if (!v.isOk()) {
        pair.live = false;
        // No worker at all is the single-node server, not news.
        if (v.code != .unsupported) say(h, .warn, "[inkling2] tensor-parallel worker lost at layer {d}: the rest runs on one node\n", .{tp.layer});
        return null;
    }
    if (tp.layer == 0) say(h, .info, "[dist] inkling2 experts engaged: {d} rows, {d} of {d} pairs on the worker\n", .{ s, p1, n });
    const tm = &pair.timing;
    tm.n += 1;
    tm.sync += t1 - t0;
    tm.host += t2 - t1;
    tm.wait += nowUs() - t2;
    if (tm.n == TIMING_LINE) {
        say(h, .debug, "[inkling2] tp rank 0: {d} exchanges of {d} rows, routing sync {d} ms, split {d} ms, send+wait {d} ms\n", .{ tm.n, s, @divTrunc(tm.sync, 1000), @divTrunc(tm.host, 1000), @divTrunc(tm.wait, 1000) });
        tm.* = .{};
    }
    const dt = ops.dtype(x);
    const sums = try g.reshape(try g.astype(try g.adopt(.{ .ctx = theirs.ctx }), dt), &.{ s, 1, d });
    const z = try g.zeros(&.{ 1, 1, d }, dt);
    const dn = if (mine) |y| try g.concat(&.{ try g.astype(y, dt), sums, z }, 0) else try g.concat(&.{ sums, z }, 0);
    const kp: c_int = @intCast(k + 1);
    return .{
        .dn = dn,
        .sel = try g.zeros(&.{@intCast(p0 + t + 1)}, .uint32),
        .inverse = try g.fromSlice(u32, inv, &.{s * kp}),
        .weights = try g.fromSlice(f32, wts, &.{ 1, s, kp }),
        .picks = kp,
    };
}

fn evalAll(xs: []const Array) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

fn queue(xs: []const Array) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_async_eval(v));
}

/// The worker's part of layer `layer`'s experts: rows `x` `[t, d]`, its pairs `pairs` `[p, 2]` u32 (row, expert;
/// sorted by expert) and their weights `[p]` f32; `[t, d]` f32 (`DistHooks.expert_share`).
pub fn expertShare(ls: []const layers.Layer, cfg: *const Config, s: mlx.mlx_stream, layer: u32, x: abi.Array, pairs: abi.Array, weights: abi.Array, out: *abi.Array) abi.Verdict {
    if (layer >= ls.len) return .{ .code = .failed, .why = abi.Str.of("expert share past the last layer") };
    var g = Graph.init(gpa, s);
    defer g.deinit();
    const y = shareRows(&g, &ls[layer], cfg, .{ .ctx = x.ctx }, .{ .ctx = pairs.ctx }, .{ .ctx = weights.ctx }) catch |e|
        return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    out.* = .{ .ctx = g.escape(y).ctx };
    return .ok;
}

fn shareRows(g: *Graph, L: *const layers.Layer, cfg: *const Config, x: Array, pairs: Array, w: Array) !Array {
    const t = ops.shape(x)[0];
    const d = ops.shape(x)[1];
    const p = ops.shape(pairs)[0];
    const k: usize = cfg.top_k;
    const at = try g.contiguous(try g.reshape(try g.sliceAxis(pairs, 1, 0, 1), &.{p}));
    const sel = try g.contiguous(try g.reshape(try g.sliceAxis(pairs, 1, 1, 2), &.{p}));
    try evalAll(&.{ at, sel, w });
    const pu: usize = @intCast(p);
    const mine: Pairs = .{
        .at = @constCast((mlx.mlx_array_data_uint32(at) orelse return error.Inkling2TpPairs)[0..pu]),
        .sel = @constCast((mlx.mlx_array_data_uint32(sel) orelse return error.Inkling2TpPairs)[0..pu]),
        .w = @constCast((mlx.mlx_array_data_float32(w) orelse return error.Inkling2TpPairs)[0..pu]),
    };
    const pl = try places(gpa, mine, @as(usize, @intCast(t)) * k, @intCast(p), downScale(L));
    defer gpa.free(pl.inv);
    defer gpa.free(pl.w);
    const dn = try g.concat(&.{ try downs(g, L, cfg, x, at, sel), try g.zeros(&.{ 1, 1, d }, ops.dtype(x)) }, 0);
    const kc: c_int = @intCast(k);
    const unused = try g.zeros(&.{1}, .float32);
    // The sum kernel's MLP output rows; its conv and add over zeros are dropped.
    const o = try kernels.pickSum(g, dn, try g.zeros(&.{p + 1}, .uint32), try g.fromSlice(u32, pl.inv, &.{t * kc}), unused, try g.fromSlice(f32, pl.w, &.{ 1, t, kc }), try g.zeros(&.{ 1, t, d }, .float32), try g.zeros(&.{ 1, @intCast(cfg.sconv_kernel - 1), d }, .float32), L.mlp_conv, unused, kc, false);
    return g.reshape(o[1], &.{ t, d });
}

/// A line through the host's log.
fn say(h: *const host_abi.Host, level: host_abi.LogLevel, comptime fmt: []const u8, args: anytype) void {
    if (!h.log_enabled(h.ctx, level)) return;
    var buf: [192]u8 = undefined;
    h.log(h.ctx, level, host_abi.Str.of(std.fmt.bufPrint(&buf, fmt, args) catch return));
}

// ── tests ──

const testing = std.testing;
const plugin = @import("plugin.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const fake_host = @import("../../sdk/testing/fake_host.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");

// Bar: every pair lands in one list exactly once, each list sorted by expert with its row and weight; the worker's
// stays within its share and misses it by less than one expert's pairs.
test "inkling2 tp: a chunk's pairs split by expert, the worker's within its share" {
    const a = testing.allocator;
    const k = 4;
    const experts = 16;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var idx: [400 * k]u32 = undefined;
    var w: [400 * k]f32 = undefined;
    // Uneven routing: low experts picked more often.
    for (&idx, &w, 0..) |*e, *x, j| {
        e.* = @intCast(@min(r.uintLessThan(u32, experts), r.uintLessThan(u32, experts)));
        x.* = @floatFromInt(j);
    }
    const parts = try split(a, &idx, &w, experts, WORKER_SHARE);
    defer for (parts) |p| free(a, p);
    try testing.expectEqual(idx.len, parts[0].sel.len + parts[1].sel.len);
    var seen: [idx.len]bool = @splat(false);
    var most: usize = 0;
    var counts: [experts]usize = @splat(0);
    for (idx) |e| counts[e] += 1;
    for (counts) |c| most = @max(most, c);
    for (parts) |p| for (p.at, p.sel, p.w, 0..) |at, e, x, i| {
        const j: usize = @intFromFloat(x);
        try testing.expect(!seen[j]);
        seen[j] = true;
        try testing.expectEqual(j, at);
        try testing.expectEqual(idx[j], e);
        if (i > 0) try testing.expect(p.sel[i - 1] <= e);
    };
    for (parts[0].sel) |e| for (parts[1].sel) |f| try testing.expect(e != f);
    const target = @as(f32, @floatFromInt(idx.len)) * WORKER_SHARE;
    const got: f32 = @floatFromInt(parts[1].sel.len);
    try testing.expect(got <= target and got > target - @as(f32, @floatFromInt(most)));
}

/// The worker in the same process: its share of a layer, or a link that fails at `fail_at`, or drops the pairs.
const TestPair = struct {
    var worker: *anyopaque = undefined;
    var fail_at: ?u32 = null;
    var present: bool = true;
    var calls: u32 = 0;
    /// The planted fault: the worker's sum comes back zero.
    var drop: bool = false;

    fn exchange(_: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, w: abi.Array, out: *abi.Array) callconv(.c) abi.Verdict {
        if (!present) return .{ .code = .unsupported, .why = abi.Str.of("no tensor-parallel worker") };
        calls += 1;
        if (fail_at == layer) return .{ .code = .failed, .why = abi.Str.of("link down") };
        const v = plugin.table.dist.?.expert_share.?(worker, layer, x, idx, w, out);
        if (v.isOk() and drop) {
            var z = mlx.mlx_array_new();
            const sh = mlx.getShape(.{ .ctx = out.ctx });
            _ = mlx.mlx_zeros(&z, sh.ptr, sh.len, .float32, mlx.gpuStream());
            _ = mlx.mlx_array_free(.{ .ctx = out.ctx });
            out.* = .{ .ctx = z.ctx };
        }
        return v;
    }

    fn prefill(a: std.mem.Allocator, pack: *conformance.Loaded, tokens: []const u32) ![]f32 {
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        const h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        const v = plugin.table.forward(pack.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
        if (!v.isOk()) return error.ForwardRefused;
        if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
        return conformance.consumeF32(a, out.logits);
    }
};

/// The share of rows whose top token two prefills agree on, in percent.
fn agree(x: []const f32, y: []const f32) usize {
    const vocab = 60;
    var same: usize = 0;
    for (0..x.len / vocab) |r| same += @intFromBool(std.mem.indexOfMax(f32, x[r * vocab ..][0..vocab]) == std.mem.indexOfMax(f32, y[r * vocab ..][0..vocab]));
    return same * 100 / (x.len / vocab);
}

// Bar: rank 0's chunk split with the worker keeps one node's top token on 98% of rows (a GPU forward of another
// width rounds otherwise, and on this random pack a last bit flips a pick here and there, `proof_test.zig`), and a
// worker whose sums are lost must not; a chunk under TP_ROWS never crosses; no worker is one node's bytes; a worker
// lost mid-forward finishes on one node and says so.
test "inkling2 tp: rank 0's split plus the worker's share is one node's prefill; no worker is its bytes; a lost one falls back and logs" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var fh: fake_host.FakeHost = undefined;
    fh.init(.{ .allocator = a, .level = .warn });
    fh.table.expert_exchange = &TestPair.exchange;
    const name = plugin.plugin.name;
    var whole = try conformance.Loaded.openWith(io, a, name, &plugin.table, try plugin.conformanceFixture(a), .{ .gpu = true });
    defer whole.close();
    var worker = try conformance.Loaded.openWith(io, a, name, &plugin.table, try plugin.conformanceFixture(a), .{ .gpu = true, .tp_rank = 1, .tp_size = 2 });
    defer worker.close();
    var rank0 = try conformance.Loaded.openWith(io, a, name, &plugin.table, try plugin.conformanceFixture(a), .{ .gpu = true, .host = fh.host(), .tp_rank = 0, .tp_size = 2 });
    defer rank0.close();
    TestPair.worker = worker.model;
    const r0: *plugin.Loaded = @ptrCast(@alignCast(rank0.model));

    const tokens = try a.alloc(u32, TP_ROWS);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7 + 1) % 60);
    const want = try TestPair.prefill(a, &whole, tokens);

    TestPair.calls = 0;
    try testing.expect(agree(want, try TestPair.prefill(a, &rank0, tokens)) >= 98);
    try testing.expect(TestPair.calls > 0);
    TestPair.drop = true;
    try testing.expect(agree(want, try TestPair.prefill(a, &rank0, tokens)) < 98);
    TestPair.drop = false;
    TestPair.calls = 0;
    _ = try TestPair.prefill(a, &rank0, tokens[0 .. TP_ROWS - 1]);
    try testing.expectEqual(@as(u32, 0), TestPair.calls);

    TestPair.present = false;
    try testing.expectEqualSlices(f32, want, try TestPair.prefill(a, &rank0, tokens));
    try testing.expectEqual(@as(usize, 0), fh.lines);
    TestPair.present = true;
    r0.pair.live = true;

    TestPair.fail_at = 3;
    try testing.expect(agree(want, try TestPair.prefill(a, &rank0, tokens)) >= 98);
    try testing.expect(std.mem.indexOf(u8, fh.last(), "worker lost at layer 3") != null);
    TestPair.fail_at = null;
}
