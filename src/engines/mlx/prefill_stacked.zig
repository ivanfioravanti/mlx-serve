//! Admitted prompts prefilled stacked: two or more cold prompts of one model (`Caps.stacks_prompts`) forward whole in
//! one batched pass, each padded at its end to the longest. Pad rows repeat the slot's last real
//! token at the next positions; every block is causal over rows (attention, the short convs), so no real row reads
//! one, and each slot's cache is truncated back to its own rows, its conv windows and its snapshot-backoff checkpoint
//! cut from the pass's per-row capture. Each slot keeps its last row's logits (`Slot.stacked_logits`): its prefill
//! samples its first token from them and forwards nothing, all but its last token taken as restored. One forward of N spans costs about what one prompt of their rows does.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const transformer_mod = @import("transformer.zig");
const checkpoint = @import("ssm/checkpoint.zig");
const Slot = @import("slot.zig").Slot;
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const slotUncachedTail = @import("prefill_interleave.zig").slotUncachedTail;
const MAX_PAD_WASTE = @import("batching/verdict.zig").MAX_PAD_WASTE;
const SSM_SNAPSHOT_BACKOFF: u32 = @intCast(@import("generate.zig").SSM_SNAPSHOT_BACKOFF);
const prefillCli = @import("generate.zig").prefillCli;
const emptySlotCache = @import("prefill_interleave.zig").emptySlotCache;
const SSM_GRID_LOW_DENSE_STRIDE = @import("generate/prefill_chunk.zig").SSM_GRID_LOW_DENSE_STRIDE;

/// Stacked rows past which a pass stops paying: from about a thousand rows a forward's cost per row stops falling, and
/// a stack only makes the first prompt wait for the last.
pub const STACK_ROWS = 1024;
/// A shorter prompt is not worth a pass.
const MIN_SPAN = 16;
pub const MAX_SLOTS = 8;

comptime {
    // A stacked span files only its backoff checkpoint: no dense low anchor may fall inside one.
    std.debug.assert(STACK_ROWS < SSM_GRID_LOW_DENSE_STRIDE);
}

/// Whether a span of up to `STACK_ROWS` rows crosses no stride checkpoint the chunk loop would file: the stride is
/// never under the prefill chunk (`effectiveSsmCheckpointStride`), so a chunk of at least the stack's rows is enough.
pub fn stridePastStack(chunk: usize) bool {
    return chunk >= STACK_ROWS;
}

/// How a slot rides a pass: its whole prompt (`cold`), or the tail its hot-cache restore leaves (`warm`).
pub const Kind = enum { none, cold, warm };

/// A slot the stacked pass may take, and the rows it forwards: a prompt of a model that stacks prompts, decoded
/// plainly enough to batch, cold or restored with a tail of `MIN_SPAN` rows or more.
fn stackable(slot: *const Slot) Item {
    const none: Item = .{ .kind = .none, .span = 0, .model = 0 };
    const cfg = slot.model.config orelse return none;
    const pb = cfg.plugin orelse return none;
    // The settle cuts each slot's windows from the pass's conv-input capture.
    if (!pb.parsed.caps.stacks_prompts or !pb.parsed.caps.captures_conv_input) return none;
    if (slot.vision_embeddings != null or slot.media.len > 0 or slot.logprobs_n > 0 or slot.prompt_logprobs) return none;
    if (slot.sampling.constraint != null or slot.sampling.penalized()) return none;
    const total = slot.full_prompt.len;
    const tail = slotUncachedTail(@constCast(slot));
    if (tail < MIN_SPAN + 1 or tail > STACK_ROWS) return none;
    return .{ .kind = if (tail == total) .cold else .warm, .span = tail, .model = @intFromPtr(slot.model) };
}

/// A candidate slot: its kind, the rows it forwards and its model.
pub const Item = struct { kind: Kind, span: usize, model: usize };

pub const Pick = struct { n: usize, width: u32, warm: bool = false, left_out: usize = 0, longest_left: usize = 0 };

/// PURE: which of `items` ride one pass (their indices in `out`) and its padded width: the first candidate's kind and
/// model, at most `MAX_SLOTS`, stacked rows within `STACK_ROWS` and padding within `MAX_PAD_WASTE` of their real rows.
/// A candidate that would break a bound prefills alone (`left_out`); one of the other kind waits for its own pass.
pub fn pickItems(items: []const Item, out: *[MAX_SLOTS]usize) Pick {
    var p: Pick = .{ .n = 0, .width = 0 };
    var first: ?Item = null;
    var real: usize = 0;
    for (items, 0..) |it, k| {
        if (it.kind == .none) continue;
        if (first) |f| if (it.kind != f.kind or it.model != f.model) continue;
        const width = @max(p.width, it.span);
        const fits = p.n < MAX_SLOTS and (p.n + 1) * width <= STACK_ROWS and
            @as(f64, @floatFromInt((p.n + 1) * width)) <= MAX_PAD_WASTE * @as(f64, @floatFromInt(real + it.span));
        if (!fits) {
            p.left_out += 1;
            p.longest_left = @max(p.longest_left, it.span);
            continue;
        }
        if (first == null) first = it;
        out[p.n] = k;
        p.n += 1;
        p.width = @intCast(width);
        real += it.span;
    }
    p.warm = if (first) |f| f.kind == .warm else false;
    if (p.n < 2) p.width = 0;
    return p;
}

/// `pickItems` over the admitted slots; the stack's slots in `out`.
pub fn pick(slots: []const *Slot, out: *[MAX_SLOTS]*Slot) Pick {
    if (!stridePastStack(prefillCli().override) or slots.len > 64) return .{ .n = 0, .width = 0 };
    var items: [64]Item = undefined;
    for (slots, items[0..slots.len]) |s, *it| it.* = stackable(s);
    var idx: [MAX_SLOTS]usize = undefined;
    const p = pickItems(items[0..slots.len], &idx);
    for (idx[0..p.n], out[0..p.n]) |k, *o| o.* = slots[k];
    return p;
}

/// One slot after a pass of `width` rows from `base` (its restored rows; 0 cold): its cache truncated to `base + span`
/// rows, its conv windows those after them, cut from the pass's capture, and (`cp_at` > 0, absolute) the checkpoint
/// there.
pub fn settleSpan(a: std.mem.Allocator, cache: *transformer_mod.KVCache, entries: ?[]transformer_mod.SSMCacheEntry, base: u32, span: u32, width: u32, cp_at: u32, s: mlx.mlx_stream) !?checkpoint.SSMCheckpoint {
    try cache.truncate(base + span, s);
    const ents = entries orelse return null;
    defer for (ents) |*e| transformer_mod.ssmFreeSpecCapture(e);
    var cp: ?checkpoint.SSMCheckpoint = null;
    errdefer if (cp) |*c| c.deinit(a);
    if (cp_at > base) cp = try checkpoint.captureSsmCheckpointAt(a, ents, cp_at, cp_at - base, width, s);
    var at = try checkpoint.captureSsmCheckpointAt(a, ents, base + span, span, width, s);
    defer at.deinit(a);
    for (ents, at.layers) |*e, *l| try transformer_mod.ssmRestore(e, l);
    return cp;
}

/// Forwards each of `slots`' prompts whole (or, `warm`, the tail each one's hot-cache restore leaves, restored here by
/// share) in one batched pass of `width` rows a slot, into the slot's own cache and recurrent entries. Their drafters
/// stay off for the request: the pass builds no assistant context. On any failure every slot is back as it was
/// admitted (`abandon`) and prefills alone.
pub fn run(slots: []const *Slot, width: u32, warm: bool, left_out: usize, longest_left: usize) !void {
    var drafters: [MAX_SLOTS]bool = undefined;
    for (slots, 0..) |s, i| drafters[i] = s.enable_drafter;
    errdefer for (slots, 0..) |s, i| abandon(s, drafters[i]);
    const xfm: *transformer_mod.Transformer = slots[0].model.transformer.?;
    var ctxs: [MAX_SLOTS]*ForwardCtx = undefined;
    var offs: [MAX_SLOTS]u32 = @splat(0);
    // A restored tail runs behind its own prefix: each slot's held state and KV, its position the restored rows.
    if (warm) for (slots, 0..) |s, i| {
        const hc = s.prefix_cache orelse return error.StackedTailMoved;
        const lk = try hc.lookupAndRestoreWithMedia(&s.cache, &s.moe_seq_offset, s.ssm_entries, xfm.s, s.full_prompt, s.has_tools, s.media, null, null, null, s.skip_prefix_cache or s.cache_variant != 0);
        const tail = s.full_prompt.len - @min(lk.matched, s.full_prompt.len);
        if (lk.matched == 0 or tail < 2 or tail > width or !holdsState(s)) return error.StackedTailMoved;
        offs[i] = @intCast(lk.matched);
        s.restored_entry = lk.entry_id;
    };
    const ids = try std.heap.page_allocator.alloc(i32, slots.len * width);
    defer std.heap.page_allocator.free(ids);
    var real: usize = 0;
    for (slots, 0..) |s, i| {
        s.ctx.cache = &s.cache;
        s.ctx.moe_seq_offset = &s.moe_seq_offset;
        s.ctx.ssm_entries = s.ssm_entries;
        s.ctx.vision_embeddings = null;
        s.ctx.capture_hidden = null;
        s.ctx.kv_attn_fused = s.kv_attn_fused;
        ctxs[i] = &s.ctx;
        const tail = s.full_prompt[offs[i]..];
        real += tail.len;
        const row = ids[i * width ..][0..width];
        for (row, 0..) |*d, k| d.* = @intCast(tail[@min(k, tail.len - 1)]);
    }
    const shape = [_]c_int{ @intCast(slots.len), @intCast(width) };
    const toks = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(toks);
    var hl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hl);
    var ha = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ha);
    const logits = try xfm.forwardMoeBatchedVerify(toks, ctxs[0..slots.len], offs[0..slots.len], &hl, &ha);
    defer _ = mlx.mlx_array_free(logits);
    for (slots, 0..) |s, i| {
        if (i == 1 and env.on(.STACKED_PREFILL_FAIL)) return error.StackedPrefillDiag;
        const total: u32 = @intCast(s.full_prompt.len);
        const span = total - offs[i];
        // The checkpoint a prefill of this prompt files: the snapshot backoff before its last token.
        const want_cp = s.prefix_cache != null and s.model.ssm_checkpoint_stride > 0 and total - 1 > SSM_SNAPSHOT_BACKOFF;
        s.stacked_checkpoint = try settleSpan(s.allocator, &s.cache, s.ssm_entries, offs[i], span, width, if (want_cp) total - 1 - SSM_SNAPSHOT_BACKOFF else 0, xfm.s);
        s.stacked_logits = try lastRow(logits, i, span, xfm.s);
        s.moe_seq_offset = total;
        s.stacked_rows = total - 1;
    }
    for (slots) |s| s.enable_drafter = false;
    log.info("[prefill] stacked {d} {s} in one forward ({d} rows a slot, {d} real of {d}); their drafters stay off this request\n", .{ slots.len, if (warm) "restored tails" else "prompts", width, real, slots.len * width });
    if (left_out > 0) log.info("[prefill] {d} prompt(s) prefill alone past the stack's bounds (longest {d} rows)\n", .{ left_out, longest_left });
}

/// Whether every recurrent entry of `s` holds a restored conv state (a tail runs behind it).
fn holdsState(s: *const Slot) bool {
    const ents = s.ssm_entries orelse return true;
    for (ents) |e| if (!e.initialized or e.conv_state.ctx == null) return false;
    return true;
}

/// `slots` with the stacked ones (their last-row logits held) first, each group in its order.
pub fn stackedFirst(slots: []*Slot) void {
    var n: usize = 0;
    for (0..slots.len) |i| if (slots[i].stacked_logits != null) {
        const s = slots[i];
        std.mem.copyBackwards(*Slot, slots[n + 1 .. i + 1], slots[n..i]);
        slots[n] = s;
        n += 1;
    };
}

/// Slot `i`'s logits at its row `span - 1` of the pass's `[N, width, vocab]`, as its own `[1, 1, vocab]` buffer.
fn lastRow(logits: mlx.mlx_array, i: usize, span: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    const at: c_int = @intCast(i);
    const row: c_int = @intCast(span - 1);
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_slice(&view, logits, &[_]c_int{ at, row, 0 }, 3, &[_]c_int{ at + 1, row + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var own = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(own);
    try mlx.check(mlx.mlx_contiguous(&own, view, false, s));
    try mlx.check(mlx.mlx_array_eval(own));
    return own;
}

/// A slot whose stacked pass failed, back as it was admitted: the pass's capture freed, its cache and recurrent entries
/// a fresh slot's, its drafter as it asked.
pub fn abandon(s: *Slot, drafter: bool) void {
    if (s.ssm_entries) |ents| for (ents) |*e| transformer_mod.ssmFreeSpecCapture(e);
    emptySlotCache(s);
    s.restored_entry = 0;
    s.stacked_rows = 0;
    if (s.stacked_checkpoint) |*cp| cp.deinit(s.allocator);
    s.stacked_checkpoint = null;
    if (s.stacked_logits) |lg| _ = mlx.mlx_array_free(lg);
    s.stacked_logits = null;
    s.enable_drafter = drafter;
}

test "pickItems: a stack is all cold prompts or all restored tails of one model, within the rows, slots and padding" {
    const t = std.testing;
    var out: [MAX_SLOTS]usize = undefined;
    // Cold first: the warm tails are left for their own stack, the other model's prompt prefills alone.
    var p = pickItems(&.{
        .{ .kind = .cold, .span = 81, .model = 1 },
        .{ .kind = .warm, .span = 100, .model = 1 },
        .{ .kind = .cold, .span = 90, .model = 1 },
        .{ .kind = .cold, .span = 85, .model = 2 },
        .{ .kind = .none, .span = 0, .model = 1 },
    }, &out);
    try t.expectEqual(@as(usize, 2), p.n);
    try t.expect(!p.warm);
    try t.expectEqualSlices(usize, &.{ 0, 2 }, out[0..2]);
    try t.expectEqual(@as(u32, 90), p.width);
    // Warm first: four restored tails behind one prefix stack, a cold one waits for its own pass.
    p = pickItems(&.{
        .{ .kind = .warm, .span = 100, .model = 1 },
        .{ .kind = .cold, .span = 81, .model = 1 },
        .{ .kind = .warm, .span = 104, .model = 1 },
        .{ .kind = .warm, .span = 97, .model = 1 },
        .{ .kind = .warm, .span = 101, .model = 1 },
    }, &out);
    try t.expectEqual(@as(usize, 4), p.n);
    try t.expect(p.warm);
    try t.expectEqualSlices(usize, &.{ 0, 2, 3, 4 }, out[0..4]);
    try t.expectEqual(@as(u32, 104), p.width);
    // The rows bound: eight tails of 200 rows stack five (5 x 200 <= 1024), the rest prefill alone.
    var many: [8]Item = undefined;
    for (&many) |*it| it.* = .{ .kind = .warm, .span = 200, .model = 1 };
    p = pickItems(&many, &out);
    try t.expectEqual(@as(usize, 5), p.n);
    try t.expectEqual(@as(usize, 3), p.left_out);
    // One stackable slot is no pass (the caller stacks from two).
    p = pickItems(&.{.{ .kind = .warm, .span = 100, .model = 1 }}, &out);
    try t.expect(p.n < 2 and p.width == 0);
}

test "stridePastStack: a stacked span crosses no stride checkpoint unless the prefill chunk is under the stack" {
    try std.testing.expect(stridePastStack(8192));
    try std.testing.expect(stridePastStack(STACK_ROWS));
    try std.testing.expect(!stridePastStack(512));
}

test "stackedFirst: the stacked slots lead, each group keeping its order" {
    var slots: [5]Slot = undefined;
    const held = [_]bool{ false, true, false, true, true };
    var order: [5]*Slot = undefined;
    for (&slots, &order, held) |*s, *o, h| {
        s.stacked_logits = if (h) .{ .ctx = @ptrFromInt(0x1000) } else null;
        o.* = s;
    }
    stackedFirst(&order);
    const want = [_]usize{ 1, 3, 4, 0, 2 };
    for (order, want) |o, w| try std.testing.expectEqual(&slots[w], o);
}
