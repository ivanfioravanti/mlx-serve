//! Batched decode on the served pack (`INKLING2_PACK`): a one-slot tick is the solo decode step, and a slot's row
//! in an N-slot tick depends on its own state only (its KV, its 48 visits' conv windows, its position).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const env = @import("../../env.zig");
const model = @import("../../sdk/config/model.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;

const testing = std.testing;
const Array = ops.Array;

/// The pack served by its plugin; null skips.
const Served = struct {
    cfg: model.ModelConfig,
    xfm: Transformer,

    fn open() !?*Served {
        const path = env.get(.INKLING2_PACK) orelse return null;
        if (mlx.noGpuBackend() or mlx.virtualGpu()) return null;
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const p = try a.create(Served);
        errdefer a.destroy(p);
        p.cfg = try model.parseConfig(io, a, std.mem.span(path));
        errdefer p.cfg.deinit(a);
        p.xfm = try driver.load(io, a, &p.cfg, std.mem.span(path), mlx.gpuStream());
        return p;
    }

    fn close(p: *Served) void {
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        testing.allocator.destroy(p);
    }
};

/// One request: its cache, its visits' recurrent entries and its position.
const Lane = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    xfm: *Transformer,
    ctx: ForwardCtx = undefined,

    fn init(l: *Lane, x: *Transformer) !void {
        const n = x.config.num_hidden_layers;
        l.* = .{ .kv = try KVCache.init(testing.allocator, n), .ssm = try fixture.newEntries(n), .xfm = x };
        l.ctx = x.defaultCtx();
        l.ctx.cache = &l.kv;
        l.ctx.moe_seq_offset = &l.off;
        l.ctx.ssm_entries = l.ssm;
    }

    fn deinit(l: *Lane) void {
        l.kv.deinit();
        l.xfm.ssmGroupDrop(l.ssm);
        fixture.freeEntries(l.ssm);
    }

    /// Prefills `n` ids of `seed`'s sequence in chunks of at most 4096 rows.
    fn prefill(l: *Lane, n: usize, seed: u32) !void {
        const ids = try testing.allocator.alloc(i32, n);
        defer testing.allocator.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(1000 + (seed * 7919 + 37 * @as(u32, @intCast(i))) % 20000);
        var at: usize = 0;
        while (at < n) {
            const end = @min(n, at + 4096);
            var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
            defer g.deinit();
            const t = try g.fromSlice(i32, ids[at..end], &.{ 1, @intCast(end - at) });
            try ops.eval(try g.adopt(try l.xfm.forwardWith(&l.ctx, t)));
            at = end;
        }
    }

    /// The solo decode step of `tok`: its logits, owned by `g`.
    fn step(l: *Lane, g: *ops.Graph, tok: i32) !Array {
        const out = try g.adopt(try l.xfm.forwardWith(&l.ctx, try g.fromSlice(i32, &.{tok}, &.{ 1, 1 })));
        try ops.eval(out);
        return out;
    }
};

/// One batched tick of `toks` over `lanes`: each slot's logits, owned by `g`.
fn tick(g: *ops.Graph, x: *Transformer, lanes: []const *Lane, toks: []const u32, out: []Array) !void {
    var ctxs: [8]*ForwardCtx = undefined;
    var offs: [8]u32 = undefined;
    for (lanes, 0..) |l, i| {
        ctxs[i] = &l.ctx;
        offs[i] = @intCast(l.off);
    }
    const rows = try x.forwardMoeBatchedDecode(toks, ctxs[0..lanes.len], offs[0..lanes.len], null);
    defer testing.allocator.free(rows);
    for (rows, out) |r, *o| {
        o.* = try g.adopt(r);
        try ops.eval(o.*);
    }
    for (lanes) |l| l.off += 1;
}

/// Every visit's conv windows of `a` equal `b`'s, bit for bit.
fn expectSameState(a: *const Lane, b: *const Lane) !void {
    for (a.ssm, b.ssm) |ea, eb| {
        try testing.expectEqual(ea.initialized, eb.initialized);
        if (ea.initialized) try ops.expectBitEqual(.{ .ctx = ea.conv_state.ctx }, .{ .ctx = eb.conv_state.ctx });
    }
}

test "inkling2 batched: a one-slot batched tick is the slot's solo decode step, bit for bit" {
    // Bar: logits and every visit's conv windows equal the solo step's over three ticks.
    const p = (try Served.open()) orelse return error.SkipZigTest;
    defer p.close();
    var solo: Lane = undefined;
    try solo.init(&p.xfm);
    defer solo.deinit();
    var one: Lane = undefined;
    try one.init(&p.xfm);
    defer one.deinit();
    try solo.prefill(300, 1);
    try one.prefill(300, 1);
    try testing.expect(p.xfm.batchedGdnReady(&.{&one.ctx}));
    for ([_]u32{ 77, 4012, 9 }) |tok| {
        var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        const want = try solo.step(&g, @intCast(tok));
        var got: [1]Array = undefined;
        try tick(&g, &p.xfm, &.{&one}, &.{tok}, &got);
        try ops.expectBitEqual(got[0], want);
        try expectSameState(&one, &solo);
    }
}

test "inkling2 batched: a slot's row depends on its own state only, at any index, past the window and the log-scale floor" {
    // Bar: twin requests (same prompt, same tokens) batched among different neighbours at different indices give the
    // same logits and conv windows bit for bit. The twins sit at twice the log-scale floor (a factor bf16 can see); a
    // neighbour past the 1024 window, another short.
    const p = (try Served.open()) orelse return error.SkipZigTest;
    defer p.close();
    var lanes: [6]Lane = undefined;
    const lens = [_]usize{ 17000, 1100, 40, 17000, 500, 2000 };
    const seeds = [_]u32{ 5, 2, 3, 5, 4, 6 };
    for (&lanes) |*l| try l.init(&p.xfm);
    defer for (&lanes) |*l| l.deinit();
    for (&lanes, lens, seeds) |*l, n, sd| try l.prefill(n, sd);
    const first = [_]*Lane{ &lanes[0], &lanes[1], &lanes[2] };
    const second = [_]*Lane{ &lanes[4], &lanes[5], &lanes[3] };
    for ([_][2]u32{ .{ 77, 900 }, .{ 4012, 31 }, .{ 9, 15000 } }) |t| {
        var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        var a: [3]Array = undefined;
        var b: [3]Array = undefined;
        try tick(&g, &p.xfm, &first, &.{ t[0], t[1], t[1] + 1 }, &a);
        try tick(&g, &p.xfm, &second, &.{ t[1] + 2, t[1] + 3, t[0] }, &b);
        try ops.expectBitEqual(b[2], a[0]);
        try expectSameState(&lanes[3], &lanes[0]);
    }
}
