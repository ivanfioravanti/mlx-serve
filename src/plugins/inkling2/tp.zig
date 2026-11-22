//! Inkling 2's experts across a tensor-parallel pair (`--dist tensor`): each rank holds a share of every expert's
//! intermediate units, rows of gate_up (whole gate and up pairs) and the matching input columns of down. Down is
//! linear in those units, so the ranks' f32 weighted sums add up to the whole MLP's. Rank 0 keeps its full banks for
//! decode, short chunks and a request the worker left; the worker keeps only its share.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const host_abi = @import("../../abi/host.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const layers = @import("layers.zig");
const kernels = @import("kernels.zig");
const Config = @import("config.zig").Config;

const Graph = ops.Graph;
const Array = ops.Array;

/// Prefill chunks from this many rows split their experts with the worker: past where the split overtakes one node.
/// Decode, verify rounds and short spans never cross the wire.
pub const TP_ROWS = 2048;

/// Rank 0's link to the worker: the host whose `expert_exchange` reaches it, and whether it still answers (a failed
/// exchange leaves the rest on this node; the server restarts to pair again).
pub const Pair = struct {
    host: *const host_abi.Host,
    live: bool = true,
};

/// One layer's share of every expert; `pair` is set on rank 0.
pub const Share = struct {
    gate_up: Bank,
    down: Bank,
    layer: u32,
    pair: ?*Pair = null,
};

/// `L`'s banks cut to `rank`'s share of `size`, as new evaluated arrays (owned by the caller). Refused where a share
/// would split a quant group or a gate/up pair.
pub fn cut(L: *const layers.Layer, g: *Graph, rank: c_int, size: c_int, layer: u32) !Share {
    // A dense bank is stored `[E, in, out]`, a quantized one `[E, out, in]`.
    const dense = L.down.s.ctx == null;
    const rows = ops.shape(L.gate_up.w)[if (dense) 2 else 1];
    if (@rem(rows, 2 * size) != 0) return error.ExpertShareOffPair;
    const part = @divExact(@divExact(rows, 2), size);
    const bits: c_int = @intCast(L.down.qp.bits);
    const gs: c_int = @intCast(L.down.qp.group_size);
    if (!dense and (@rem(part, gs) != 0 or @rem(part * bits, 32) != 0)) return error.ExpertShareOffGroup;
    const gu = [2]c_int{ 2 * part * rank, 2 * part * (rank + 1) };
    // Down's packed words (a dense bank's rows) and group scales along its input axis.
    const words = if (dense) part else @divExact(part * bits, 32);
    const groups = if (dense) 0 else @divExact(part, gs);
    var out: Share = .{ .gate_up = L.gate_up, .down = L.down, .layer = layer };
    out.gate_up.w = try slice(g, L.gate_up.w, if (dense) 2 else 1, gu[0], gu[1]);
    out.gate_up.s = if (L.gate_up.s.ctx != null) try slice(g, L.gate_up.s, 1, gu[0], gu[1]) else .{};
    out.gate_up.b = if (L.gate_up.b.ctx != null) try slice(g, L.gate_up.b, 1, gu[0], gu[1]) else .{};
    out.down.w = try slice(g, L.down.w, if (dense) 1 else 2, words * rank, words * (rank + 1));
    out.down.s = if (!dense) try slice(g, L.down.s, 2, groups * rank, groups * (rank + 1)) else .{};
    out.down.b = if (L.down.b.ctx != null) try slice(g, L.down.b, 2, groups * rank, groups * (rank + 1)) else .{};
    return out;
}

/// A contiguous evaluated copy of `x[.., lo:hi, ..]` on `axis`, escaped from `g`.
fn slice(g: *Graph, x: Array, axis: usize, lo: c_int, hi: c_int) !Array {
    const y = try g.contiguous(try g.sliceAxis(x, axis, lo, hi));
    try ops.eval(y);
    return g.escape(y);
}

/// The weighted f32 sum `[1, S, D]` of the rows `x` `[1, S, D]` through `share`'s units of their experts `idx`
/// `[1, S, k]` with weights `[1, S, k]`: one rank's part of the MLP, as the gather path of `kernels.experts` runs it.
pub fn partial(g: *Graph, L: *const layers.Layer, cfg: *const Config, share: *const Share, x: Array, idx: Array, weights: Array) !Array {
    const s = ops.shape(x)[1];
    const d = ops.shape(x)[2];
    const k: c_int = @intCast(cfg.top_k);
    const p = try layers.pairs(g, cfg, x, idx);
    const sorted = p.inverse != null;
    const caps = try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const scaled = L.gate_up_scale.ctx != null;
    const act = try kernels.activation(g, try layers.gather(g, p.rows, &share.gate_up, p.sel, sorted), p.sel, if (scaled) L.gate_up_scale else caps, caps, scaled);
    var dn = try g.reshape(try g.astype(try layers.gather(g, act, &share.down, p.sel, sorted), .float32), &.{ s * k, d });
    if (L.down_scale.ctx != null) dn = try g.mul(dn, try g.expandDims(try g.take(L.down_scale, p.sel, 0), -1));
    if (p.inverse) |inverse| dn = try g.take(dn, inverse, 0);
    return g.sum(try g.mul(try g.reshape(dn, &.{ 1, s, k, d }), try g.expandDims(try g.astype(weights, .float32), -1)), 2, false);
}

/// Rank 0, a prefill chunk's MLP: its share queued on the GPU, the worker's through the host, summed in f32, in
/// `x`'s dtype `[1, S, D]`; null = the single-node MLP (a short chunk, no pair, or the worker gone).
pub fn exchange(g: *Graph, L: *const layers.Layer, cfg: *const Config, x: Array, idx: Array, weights: Array) !?Array {
    const share = &(L.share orelse return null);
    const pair = share.pair orelse return null;
    const s = ops.shape(x)[1];
    if (!pair.live or s < TP_ROWS) return null;
    const h = pair.host;
    const send = host_abi.expertExchangeOf(h) orelse return null;
    const d = ops.shape(x)[2];
    const k: c_int = @intCast(cfg.top_k);
    const flat = try g.reshape(x, &.{ s, d });
    const routes = try g.astype(try g.reshape(idx, &.{ s, k }), .uint32);
    const w32 = try g.astype(try g.reshape(weights, &.{ s, k }), .float32);
    // The rows go out as soon as they exist: queued with this rank's share, the send waits for the whole batch.
    try queue(&.{ flat, routes, w32 });
    const mine = try partial(g, L, cfg, share, x, idx, weights);
    try queue(&.{mine});
    var theirs: host_abi.Array = .{};
    const v = send(h.ctx, share.layer, .{ .ctx = flat.ctx }, .{ .ctx = routes.ctx }, .{ .ctx = w32.ctx }, &theirs);
    if (!v.isOk()) {
        pair.live = false;
        // No worker at all is the single-node server, not news.
        if (v.code != .unsupported) say(h, .warn, "[inkling2] tensor-parallel worker lost at layer {d}: the rest runs on one node\n", .{share.layer});
        return null;
    }
    if (share.layer == 0) say(h, .info, "[dist] inkling2 experts engaged: {d} rows\n", .{s});
    const sum = try g.add(mine, try g.astype(try g.reshape(try g.adopt(.{ .ctx = theirs.ctx }), &.{ 1, s, d }), .float32));
    return try g.astype(sum, ops.dtype(x));
}

fn queue(xs: []const Array) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_async_eval(v));
}

/// The worker's share of layer `layer`'s experts for rank 0's rows `x` `[t, d]`, routes `idx` `[t, k]` and weights
/// `[t, k]`: `[t, d]` f32 (`DistHooks.expert_share`).
pub fn expertShare(ls: []const layers.Layer, cfg: *const Config, s: mlx.mlx_stream, layer: u32, x: abi.Array, idx: abi.Array, weights: abi.Array, out: *abi.Array) abi.Verdict {
    if (layer >= ls.len) return .{ .code = .failed, .why = abi.Str.of("expert share past the last layer") };
    const L = &ls[layer];
    const share = &(L.share orelse return .{ .code = .unsupported, .why = abi.Str.of("loaded without a tensor-parallel rank") });
    var g = Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const y = shareRows(&g, L, cfg, share, .{ .ctx = x.ctx }, .{ .ctx = idx.ctx }, .{ .ctx = weights.ctx }) catch |e|
        return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    out.* = .{ .ctx = g.escape(y).ctx };
    return .ok;
}

fn shareRows(g: *Graph, L: *const layers.Layer, cfg: *const Config, share: *const Share, x: Array, idx: Array, weights: Array) !Array {
    const t = ops.shape(x)[0];
    const d = ops.shape(x)[1];
    const k = ops.shape(idx)[1];
    const y = try partial(g, L, cfg, share, try g.reshape(x, &.{ 1, t, d }), try g.reshape(idx, &.{ 1, t, k }), try g.reshape(weights, &.{ 1, t, k }));
    return g.reshape(y, &.{ t, d });
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

/// The worker in the same process: its share of a layer, or a link that fails at `fail_at`.
const TestPair = struct {
    var worker: *anyopaque = undefined;
    var fail_at: ?u32 = null;
    var present: bool = true;
    var calls: u32 = 0;

    fn exchange(_: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, w: abi.Array, out: *abi.Array) callconv(.c) abi.Verdict {
        if (!present) return .{ .code = .unsupported, .why = abi.Str.of("no tensor-parallel worker") };
        calls += 1;
        if (fail_at == layer) return .{ .code = .failed, .why = abi.Str.of("link down") };
        return plugin.table.dist.?.expert_share.?(worker, layer, x, idx, w, out);
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

// Bar: rank 0's chunk through both shares gives one node's logits to f32 summation order; a chunk under TP_ROWS never
// crosses; no worker is one node's bytes; a worker lost mid-forward finishes on one node and says so once.
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
    // The worker holds only its share of each bank.
    const held: *plugin.Loaded = @ptrCast(@alignCast(worker.model));
    const full: *plugin.Loaded = @ptrCast(@alignCast(whole.model));
    try testing.expectEqual(@divExact(ops.shape(full.layers[0].gate_up.w)[2], 2), ops.shape(held.layers[0].gate_up.w)[2]);

    const tokens = try a.alloc(u32, TP_ROWS);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7 + 1) % 60);
    const want = try TestPair.prefill(a, &whole, tokens);
    // A GPU forward of another width rounds otherwise, and on this random pack a last bit flips an expert pick here
    // and there (`proof_test.zig`): the bar is each row's top token, not its logits.
    const near = struct {
        fn check(x: []const f32, y: []const f32) !void {
            const vocab = 60;
            var same: usize = 0;
            for (0..x.len / vocab) |r| same += @intFromBool(std.mem.indexOfMax(f32, x[r * vocab ..][0..vocab]) == std.mem.indexOfMax(f32, y[r * vocab ..][0..vocab]));
            try testing.expect(same * 100 >= 98 * (x.len / vocab));
        }
    };
    const visits = held.base.weights.body.cfg.visits();

    TestPair.calls = 0;
    try near.check(want, try TestPair.prefill(a, &rank0, tokens));
    try testing.expectEqual(@as(u32, @intCast(visits)), TestPair.calls);
    TestPair.calls = 0;
    _ = try TestPair.prefill(a, &rank0, tokens[0 .. TP_ROWS - 1]);
    try testing.expectEqual(@as(u32, 0), TestPair.calls);

    const r0: *plugin.Loaded = @ptrCast(@alignCast(rank0.model));
    TestPair.present = false;
    try testing.expectEqualSlices(f32, want, try TestPair.prefill(a, &rank0, tokens));
    try testing.expectEqual(@as(usize, 0), fh.lines);
    TestPair.present = true;
    r0.pair.live = true;

    TestPair.fail_at = 3;
    try near.check(want, try TestPair.prefill(a, &rank0, tokens));
    try testing.expect(std.mem.indexOf(u8, fh.last(), "worker lost at layer 3") != null);
    TestPair.fail_at = null;
}

// Bar: the two ranks' shares of a layer's experts add up to the whole bank's weighted sum (the bank as one share) to
// f32 summation order, at a chunk of rows; a worker cut as rank 0 must not.
test "inkling2 tp: two shares of every expert's units sum to the whole MLP" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pack = try conformance.Loaded.openWith(testing.io, a, plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(a), .{ .gpu = true });
    defer pack.close();
    const m: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
    const cfg = &m.base.weights.body.cfg;
    const L = &m.layers[1];
    var g = Graph.init(testing.allocator, m.base.s);
    defer g.deinit();
    const key = try g.call(mlx.mlx_random_key, .{@as(u64, 7)});
    const x = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, 40, @intCast(cfg.hidden_size) }, @as(usize, 3), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key });
    const r = try layers.route(&g, L, cfg, x);
    const shares = [_]Share{ try cut(L, &g, 0, 1, 1), try cut(L, &g, 0, 2, 1), try cut(L, &g, 1, 2, 1) };
    for (shares) |sh| for ([_]Array{ sh.gate_up.w, sh.gate_up.s, sh.gate_up.b, sh.down.w, sh.down.s, sh.down.b }) |w| if (w.ctx != null) {
        _ = try g.adopt(w);
    };
    const want = try partial(&g, L, cfg, &shares[0], x, r.idx, r.weights);
    const scale = try g.addScalar(try g.call(mlx.mlx_max, .{ try g.abs(want), false }), 1e-6);
    for ([_]struct { Array, bool }{
        .{ try g.add(try partial(&g, L, cfg, &shares[1], x, r.idx, r.weights), try partial(&g, L, cfg, &shares[2], x, r.idx, r.weights)), true },
        .{ try g.add(try partial(&g, L, cfg, &shares[1], x, r.idx, r.weights), try partial(&g, L, cfg, &shares[1], x, r.idx, r.weights)), false },
    }) |c| {
        const q = try g.div(try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(c[0], want)), false }), scale);
        try ops.eval(q);
        var v: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&v, q));
        if (c[1]) try testing.expect(v <= 1e-5) else try testing.expect(v > 1e-2);
    }
}
