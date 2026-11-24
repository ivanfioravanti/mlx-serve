//! Inkling 2 served through the engine's plugin driver on its pack (`INKLING2_PACK`): a prompt restored from the
//! prefix cache's checkpoints, as the scheduler restores one, decodes from the logits the cold prompt decodes from.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const HotPrefixCache = @import("../../engines/mlx/cache/prefix_cache.zig").HotPrefixCache;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const TokenLogprob = @import("../../sdk/request/logprob.zig").TokenLogprob;

const testing = std.testing;
const TOP = 20;

/// The first token's top logprobs of `prompt[from..]` after `from` rows were restored, as the scheduler's generator
/// takes them (stride-128 checkpoints, greedy); with `hc`, the prompt's rows and checkpoints are committed to it.
fn firstLogprobs(xfm: *Transformer, prompt: []const u32, from: usize, hc: ?*HotPrefixCache) ![TOP]TokenLogprob {
    const a = testing.allocator;
    var tok: Tokenizer = undefined;
    var gen = try Generator.initWithOptions(testing.io, a, xfm, &tok, prompt[from..], 1, .{ .temperature = 0.0 }, &.{}, .{ .ssm_checkpoint_stride = 128, .ssm_checkpoint_pos_offset = from, .logprobs_n = TOP });
    defer gen.deinit(a);
    var out: [TOP]TokenLogprob = undefined;
    @memcpy(&out, gen.pending_logprob.?.top_logprobs[0..TOP]);
    if (hc) |c| {
        const cps = gen.takeSsmCheckpoints();
        _ = try c.commitWithSsm(&xfm.cache, prompt, false, if (cps.len > 0) cps else null, null, null);
    }
    return out;
}

/// The same prompt restored from `hc`: the rows the lookup matched come from the cache, the rest are forwarded.
fn warmLogprobs(xfm: *Transformer, prompt: []const u32, hc: *HotPrefixCache) !struct { lp: [TOP]TokenLogprob, matched: usize } {
    try xfm.resetCache();
    const hit = try hc.lookupAndRestore(&xfm.cache, &xfm.moe_seq_offset, xfm.ssm_entries, xfm.s, prompt, false, &.{}, null, null);
    try testing.expect(hit.matched > 0 and hit.matched < prompt.len);
    return .{ .lp = try firstLogprobs(xfm, prompt, hit.matched, null), .matched = hit.matched };
}

// Bar: an exact repeat restored from the cold prompt's checkpoints takes the first token's top-20 logprobs the cold
// prompt takes, bit for bit, at two prompt lengths.
test "inkling2 pack: an exact repeat restored from the prefix cache decodes from the cold prompt's logits bit for bit (INKLING2_PACK)" {
    const dir = env.get(.INKLING2_PACK) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var cfg = try model.parseConfig(testing.io, a, std.mem.span(dir));
    defer cfg.deinit(a);
    var xfm = try driver.load(testing.io, a, &cfg, std.mem.span(dir), mlx.gpuStream());
    defer xfm.deinit();
    var ids: [1430]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 50000);
    for ([_]usize{ 300, ids.len }) |n| {
        var hc = HotPrefixCache.init(a, 4);
        defer hc.deinit();
        try xfm.resetCache();
        const cold = try firstLogprobs(&xfm, ids[0..n], 0, &hc);
        const warm = try warmLogprobs(&xfm, ids[0..n], &hc);
        if (!std.mem.eql(u8, std.mem.asBytes(&cold), std.mem.asBytes(&warm.lp))) std.debug.print("[inkling2 served] {d} rows restored at {d}: first logprobs differ\n", .{ n, warm.matched });
        try testing.expectEqualSlices(u8, std.mem.asBytes(&cold), std.mem.asBytes(&warm.lp));
    }
}
