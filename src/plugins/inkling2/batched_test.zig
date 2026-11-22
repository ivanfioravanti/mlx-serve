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

    /// Prefills `n` ids of `seed`'s sequence in chunks of at most 4096 rows; from seed 100 the sequence repeats every 47
    /// ids, a prompt the model continues confidently.
    fn prefill(l: *Lane, n: usize, seed: u32) !void {
        const ids = try testing.allocator.alloc(i32, n);
        defer testing.allocator.free(ids);
        for (ids, 0..) |*v, i| v.* = if (seed >= 100) @intCast(1000 + (seed * 7919 + 37 * @as(u32, @intCast(i % 47))) % 20000) else @intCast(1000 + (seed * 7919 + 37 * @as(u32, @intCast(i))) % 20000);
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

/// Row `i` of logits `[rows, vocab]` (or `[vocab]`) on the host, f32.
fn hostRow(g: *ops.Graph, x: Array, i: c_int) ![]const f32 {
    const sh = ops.shape(x);
    const row = if (sh.len == 2) try g.sliceAxis(x, 0, i, i + 1) else x;
    const f = try g.contiguous(try g.astype(row, .float32));
    try ops.eval(f);
    return mlx.mlx_array_data_float32(f).?[0..mlx.mlx_array_size(f)];
}

fn argmax(v: []const f32) u32 {
    var best: usize = 0;
    for (v, 0..) |x, i| if (x > v[best]) {
        best = i;
    };
    return @intCast(best);
}

test "inkling2 batched: two slots decode what each decodes alone, within the routed kernels' numerics" {
    // Bar: greedy ids equal to each prompt's solo run for 16 steps, and the batched row within 0.75 of the solo
    // logits over the solo's top 10 (the MoE's multi-row kernels round apart from its one-token ones, and each side's
    // own KV and windows carry that apart step by step; repeating prompts keep greedy off near-ties).
    const p = (try Served.open()) orelse return error.SkipZigTest;
    defer p.close();
    var lanes: [4]Lane = undefined;
    for (&lanes) |*l| try l.init(&p.xfm);
    defer for (&lanes) |*l| l.deinit();
    const lens = [_]usize{ 600, 1300, 600, 1300 };
    const seeds = [_]u32{ 108, 109, 108, 109 };
    for (&lanes, lens, seeds) |*l, n, sd| try l.prefill(n, sd);
    var next = [2]u32{ 4021, 977 };
    var worst: f32 = 0;
    for (0..16) |_| {
        var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
        defer g.deinit();
        var got: [2]Array = undefined;
        try tick(&g, &p.xfm, &.{ &lanes[2], &lanes[3] }, &next, &got);
        for (0..2) |i| {
            const solo = try hostRow(&g, try lanes[i].step(&g, @intCast(next[i])), 0);
            const batched = try hostRow(&g, got[i], 0);
            try testing.expectEqual(argmax(solo), argmax(batched));
            const top = try testing.allocator.dupe(f32, solo);
            defer testing.allocator.free(top);
            std.mem.sort(f32, top, {}, std.sort.desc(f32));
            const floor = top[9];
            for (solo, batched) |a, b| if (a >= floor) {
                worst = @max(worst, @abs(a - b));
            };
            next[i] = argmax(solo);
        }
    }
    std.debug.print("[inkling2 batched] worst gap over the solo top 10: {d:.3}\n", .{worst});
    try testing.expect(worst <= 0.75);
}

// Bar: a restored tail's checkpoint from one capturing forward is the chunk loop's, every visit's windows bit for bit.
test "inkling2 served: a 140-row tail's checkpoint cut at 109 rows from one capturing forward is the chunk loop's" {
    const p = (try Served.open()) orelse return error.SkipZigTest;
    defer p.close();
    const checkpoint = @import("../../engines/mlx/ssm/checkpoint.zig");
    var chunked: Lane = undefined;
    try chunked.init(&p.xfm);
    defer chunked.deinit();
    var whole: Lane = undefined;
    try whole.init(&p.xfm);
    defer whole.deinit();
    for ([_]*Lane{ &chunked, &whole }) |l| {
        l.ctx.prefill = true;
        try l.prefill(600, 3);
    }
    var tail: [140]i32 = undefined;
    for (&tail, 0..) |*v, i| v.* = @intCast(1000 + (11 * 7919 + 37 * @as(u32, @intCast(i))) % 20000);
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    try ops.eval(try g.adopt(try p.xfm.forwardWith(&chunked.ctx, try g.fromSlice(i32, tail[0..109], &.{ 1, 109 }))));
    var want = try checkpoint.captureSsmCheckpoint(testing.allocator, chunked.ssm, 709, p.xfm.s);
    defer want.deinit(testing.allocator);
    whole.ctx.capture_ssm_seq = true;
    try ops.eval(try g.adopt(try p.xfm.forwardWith(&whole.ctx, try g.fromSlice(i32, &tail, &.{ 1, 140 }))));
    whole.ctx.capture_ssm_seq = false;
    var got = try checkpoint.captureSsmCheckpointAt(testing.allocator, whole.ssm, 709, 109, 140, p.xfm.s);
    defer got.deinit(testing.allocator);
    for (want.layers, got.layers) |w, h| {
        try testing.expectEqual(w.initialized, h.initialized);
        if (w.conv_state.ctx != null) try ops.expectBitEqual(w.conv_state, h.conv_state);
    }
}

/// `ids` of each lane as `[N, w]` through one batched forward: each lane's last-row logits `[1, vocab]`, owned by `g`.
fn stackedPrefill(g: *ops.Graph, x: *Transformer, lanes: []const *Lane, ids: []const i32, w: usize) ![8]Array {
    var ctxs: [8]*ForwardCtx = undefined;
    var offs: [8]u32 = undefined;
    for (lanes, 0..) |l, i| {
        ctxs[i] = &l.ctx;
        offs[i] = @intCast(l.off);
    }
    const toks = try g.fromSlice(i32, ids, &.{ @intCast(lanes.len), @intCast(w) });
    var hl = mlx.mlx_array_new();
    var ha = mlx.mlx_array_new();
    const logits = try g.adopt(try x.forwardMoeBatchedVerify(toks, ctxs[0..lanes.len], offs[0..lanes.len], &hl, &ha));
    _ = try g.adopt(hl);
    _ = try g.adopt(ha);
    for (lanes) |l| l.off += w;
    var out: [8]Array = undefined;
    const iw: c_int = @intCast(w);
    for (0..lanes.len) |i| {
        const ii: c_int = @intCast(i);
        out[i] = try g.reshape(try g.sliceAxis(try g.sliceAxis(logits, 0, ii, ii + 1), 1, iw - 1, iw), &.{ 1, ops.shape(logits)[2] });
    }
    try ops.evalMany(out[0..lanes.len]);
    return out;
}

test "inkling2 batched: slots' prompt spans in one forward give each span's solo prefill, within the routed kernels' numerics" {
    // Bar: each slot's last row has its solo prefill's top token, within 0.75 of its logits over their top 10 (the MoE
    // rounds the stacked rows apart from a lone span's), and leaves its conv windows within the same class.
    const p = (try Served.open()) orelse return error.SkipZigTest;
    defer p.close();
    const n = 3;
    const w = 81;
    var lanes: [2 * n]Lane = undefined;
    for (&lanes) |*l| try l.init(&p.xfm);
    defer for (&lanes) |*l| l.deinit();
    // Slots after their own restored prefixes of different lengths, then a prompt span each.
    for (0..n) |i| for ([_]usize{ i, n + i }) |li| try lanes[li].prefill(200 + 150 * i, @intCast(100 + i));
    var ids: [n * w]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + (101 * 7919 + 53 * @as(u32, @intCast(i))) % 20000);
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var stacked_lanes: [n]*Lane = undefined;
    for (&stacked_lanes, 0..) |*s, i| s.* = &lanes[n + i];
    const got = try stackedPrefill(&g, &p.xfm, &stacked_lanes, &ids, w);
    var worst: f32 = 0;
    for (0..n) |i| {
        const solo_logits = try g.adopt(try p.xfm.forwardWith(&lanes[i].ctx, try g.fromSlice(i32, ids[i * w ..][0..w], &.{ 1, w })));
        const sh = ops.shape(solo_logits);
        const solo = try hostRow(&g, try g.reshape(try g.sliceAxis(solo_logits, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] }), 0);
        const stacked = try hostRow(&g, got[i], 0);
        try testing.expectEqual(argmax(solo), argmax(stacked));
        const top = try testing.allocator.dupe(f32, solo);
        defer testing.allocator.free(top);
        std.mem.sort(f32, top, {}, std.sort.desc(f32));
        for (solo, stacked) |a, b| if (a >= top[9]) {
            worst = @max(worst, @abs(a - b));
        };
    }
    std.debug.print("[inkling2 batched] stacked prompt spans, worst gap over the solo top 10: {d:.3}\n", .{worst});
    try testing.expect(worst <= 0.75);
}
