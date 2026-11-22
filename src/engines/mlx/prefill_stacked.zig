//! Admitted prompts prefilled stacked: two or more cold prompts of one model (`Caps.stacks_prompts`) forward their
//! common span in one batched pass, then each prefills the rest of its prompt alone, the stacked rows taken as
//! restored (`Slot.stacked_rows`). One forward of N spans costs about what one prompt of their rows does, where N
//! serial prefills pay a short forward's fixed cost each.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const transformer_mod = @import("transformer.zig");
const Slot = @import("slot.zig").Slot;
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const slotUncachedTail = @import("prefill_interleave.zig").slotUncachedTail;

/// Stacked rows past which a pass stops paying: from about a thousand rows a forward's cost per row stops falling, and
/// a stack only makes the first prompt wait for the last.
pub const STACK_ROWS = 1024;
/// Rows each prompt keeps for its own prefill: the snapshot backoff's tail and its last row.
const OWN_TAIL = 32;
/// A shorter common span is not worth a pass.
const MIN_SPAN = 16;
const MAX_SLOTS = 8;

/// A slot the stacked pass may take: a cold prompt of a model that stacks prompts, decoded plainly enough to batch.
fn stackable(slot: *const Slot) bool {
    const cfg = slot.model.config orelse return false;
    const pb = cfg.plugin orelse return false;
    if (!pb.parsed.caps.stacks_prompts) return false;
    if (slot.vision_embeddings != null or slot.media.len > 0 or slot.logprobs_n > 0 or slot.prompt_logprobs) return false;
    if (slot.sampling.constraint != null or slot.sampling.penalized()) return false;
    if (slot.full_prompt.len < OWN_TAIL + MIN_SPAN or slot.full_prompt.len > STACK_ROWS) return false;
    // A prompt the hot cache would restore keeps its restore.
    return slotUncachedTail(@constCast(slot)) == slot.full_prompt.len;
}

/// The slots of `slots` that ride one pass, and its span: the first stackable slot's model, at most `MAX_SLOTS`,
/// stacked rows within `STACK_ROWS`. `n` < 2 = no pass.
pub fn pick(slots: []const *Slot, out: *[MAX_SLOTS]*Slot) struct { n: usize, span: u32 } {
    var n: usize = 0;
    var span: usize = std.math.maxInt(usize);
    for (slots) |s| {
        if (n == MAX_SLOTS or !stackable(s)) continue;
        if (n > 0 and s.model != out[0].model) continue;
        const next_span = @min(span, s.full_prompt.len - OWN_TAIL);
        if ((n + 1) * next_span > STACK_ROWS) break;
        out[n] = s;
        n += 1;
        span = next_span;
    }
    return .{ .n = n, .span = if (n >= 2) @intCast(span) else 0 };
}

/// Forwards `slots`' first `span` prompt rows in one batched pass into each slot's own cache and recurrent entries.
/// Their drafters stay off for the request: the pass builds no assistant context.
pub fn run(slots: []const *Slot, span: u32) !void {
    const xfm: *transformer_mod.Transformer = slots[0].model.transformer.?;
    var ctxs: [MAX_SLOTS]*ForwardCtx = undefined;
    var offs: [MAX_SLOTS]u32 = @splat(0);
    const ids = try std.heap.page_allocator.alloc(i32, slots.len * span);
    defer std.heap.page_allocator.free(ids);
    for (slots, 0..) |s, i| {
        s.ctx.cache = &s.cache;
        s.ctx.moe_seq_offset = &s.moe_seq_offset;
        s.ctx.ssm_entries = s.ssm_entries;
        s.ctx.vision_embeddings = null;
        s.ctx.capture_hidden = null;
        s.ctx.kv_attn_fused = s.kv_attn_fused;
        ctxs[i] = &s.ctx;
        for (s.full_prompt[0..span], ids[i * span ..][0..span]) |t, *d| d.* = @intCast(t);
    }
    const shape = [_]c_int{ @intCast(slots.len), @intCast(span) };
    const toks = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(toks);
    var hl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hl);
    var ha = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ha);
    const logits = try xfm.forwardMoeBatchedVerify(toks, ctxs[0..slots.len], offs[0..slots.len], &hl, &ha);
    _ = mlx.mlx_array_free(logits);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (slots) |s| {
        for (s.cache.entries) |*e| if (e.initialized) {
            _ = mlx.mlx_vector_array_append_value(vec, e.keys);
            _ = mlx.mlx_vector_array_append_value(vec, e.values);
        };
        if (s.ssm_entries) |ents| for (ents) |*e| if (e.conv_state.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(vec, e.conv_state);
        };
    }
    try mlx.check(mlx.mlx_eval(vec));
    for (slots) |s| {
        // The pass captured per row as a verify does; nothing reads it.
        if (s.ssm_entries) |ents| for (ents) |*e| transformer_mod.ssmFreeSpecCapture(e);
        s.moe_seq_offset = span;
        s.stacked_rows = span;
        s.enable_drafter = false;
    }
    log.info("[prefill] stacked {d} prompts x {d} rows in one forward ({d} rows); their drafters stay off this request\n", .{ slots.len, span, slots.len * span });
}
