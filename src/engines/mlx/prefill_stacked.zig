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

/// A slot the stacked pass may take: a cold prompt of a model that stacks prompts, decoded plainly enough to batch.
fn stackable(slot: *const Slot) bool {
    const cfg = slot.model.config orelse return false;
    const pb = cfg.plugin orelse return false;
    if (!pb.parsed.caps.stacks_prompts) return false;
    if (slot.vision_embeddings != null or slot.media.len > 0 or slot.logprobs_n > 0 or slot.prompt_logprobs) return false;
    if (slot.sampling.constraint != null or slot.sampling.penalized()) return false;
    if (slot.full_prompt.len < MIN_SPAN + 1 or slot.full_prompt.len > STACK_ROWS) return false;
    // A prompt the hot cache would restore keeps its restore.
    return slotUncachedTail(@constCast(slot)) == slot.full_prompt.len;
}

pub const Pick = struct { n: usize, width: u32, left_out: usize = 0, longest_left: usize = 0 };

/// The slots of `slots` that ride one pass and its padded width: the first stackable slot's model, at most `MAX_SLOTS`,
/// stacked rows within `STACK_ROWS` and padding within `MAX_PAD_WASTE` of their real rows. A slot that would break a
/// bound prefills alone. `n` < 2 = no pass.
pub fn pick(slots: []const *Slot, out: *[MAX_SLOTS]*Slot) Pick {
    var p: Pick = .{ .n = 0, .width = 0 };
    if (!stridePastStack(prefillCli().override)) return p;
    var real: usize = 0;
    for (slots) |s| {
        if (!stackable(s) or (p.n > 0 and s.model != out[0].model)) continue;
        const span = s.full_prompt.len;
        const width = @max(p.width, span);
        const fits = p.n < MAX_SLOTS and (p.n + 1) * width <= STACK_ROWS and
            @as(f64, @floatFromInt((p.n + 1) * width)) <= MAX_PAD_WASTE * @as(f64, @floatFromInt(real + span));
        if (!fits) {
            p.left_out += 1;
            p.longest_left = @max(p.longest_left, span);
            continue;
        }
        out[p.n] = s;
        p.n += 1;
        p.width = @intCast(width);
        real += span;
    }
    if (p.n < 2) p.width = 0;
    return p;
}

/// One slot after a pass of `width` rows: its cache truncated to its own `span` rows, its conv windows those after
/// them, cut from the pass's capture, and (`cp_at` > 0) the checkpoint after that many rows.
pub fn settleSpan(a: std.mem.Allocator, cache: *transformer_mod.KVCache, entries: ?[]transformer_mod.SSMCacheEntry, span: u32, width: u32, cp_at: u32, s: mlx.mlx_stream) !?checkpoint.SSMCheckpoint {
    try cache.truncate(span, s);
    const ents = entries orelse return null;
    defer for (ents) |*e| transformer_mod.ssmFreeSpecCapture(e);
    var cp: ?checkpoint.SSMCheckpoint = null;
    errdefer if (cp) |*c| c.deinit(a);
    if (cp_at > 0) cp = try checkpoint.captureSsmCheckpointAt(a, ents, cp_at, cp_at, width, s);
    var at = try checkpoint.captureSsmCheckpointAt(a, ents, span, span, width, s);
    defer at.deinit(a);
    for (ents, at.layers) |*e, *l| try transformer_mod.ssmRestore(e, l);
    return cp;
}

/// Forwards each of `slots`' prompts whole in one batched pass of `width` rows a slot, into the slot's
/// own cache and recurrent entries. Their drafters stay off for the request: the pass builds no assistant context. On
/// any failure every slot is back as it was admitted (`abandon`) and prefills alone.
pub fn run(slots: []const *Slot, width: u32, left_out: usize, longest_left: usize) !void {
    var drafters: [MAX_SLOTS]bool = undefined;
    for (slots, 0..) |s, i| drafters[i] = s.enable_drafter;
    errdefer for (slots, 0..) |s, i| abandon(s, drafters[i]);
    const xfm: *transformer_mod.Transformer = slots[0].model.transformer.?;
    var ctxs: [MAX_SLOTS]*ForwardCtx = undefined;
    var offs: [MAX_SLOTS]u32 = @splat(0);
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
        const span = s.full_prompt.len;
        real += span;
        const row = ids[i * width ..][0..width];
        for (row, 0..) |*d, k| d.* = @intCast(s.full_prompt[@min(k, span - 1)]);
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
        const span: u32 = @intCast(s.full_prompt.len);
        // The checkpoint a prefill of this prompt files: the snapshot backoff before its last token.
        const want_cp = s.prefix_cache != null and s.model.ssm_checkpoint_stride > 0 and span - 1 > SSM_SNAPSHOT_BACKOFF;
        s.stacked_checkpoint = try settleSpan(s.allocator, &s.cache, s.ssm_entries, span, width, if (want_cp) span - 1 - SSM_SNAPSHOT_BACKOFF else 0, xfm.s);
        s.stacked_logits = try lastRow(logits, i, span, xfm.s);
        s.moe_seq_offset = span;
        s.stacked_rows = span - 1;
    }
    for (slots) |s| s.enable_drafter = false;
    log.info("[prefill] stacked {d} prompts in one forward ({d} rows a slot, {d} real of {d}); their drafters stay off this request\n", .{ slots.len, width, real, slots.len * width });
    if (left_out > 0) log.info("[prefill] {d} prompt(s) prefill alone past the stack's bounds (longest {d} rows)\n", .{ left_out, longest_left });
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
    s.stacked_rows = 0;
    if (s.stacked_checkpoint) |*cp| cp.deinit(s.allocator);
    s.stacked_checkpoint = null;
    if (s.stacked_logits) |lg| _ = mlx.mlx_array_free(lg);
    s.stacked_logits = null;
    s.enable_drafter = drafter;
}

test "stridePastStack: a stacked span crosses no stride checkpoint unless the prefill chunk is under the stack" {
    try std.testing.expect(stridePastStack(8192));
    try std.testing.expect(stridePastStack(STACK_ROWS));
    try std.testing.expect(!stridePastStack(512));
}
