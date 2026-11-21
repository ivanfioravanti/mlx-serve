//! Runtime spec gates: warmups, per-draft rate, DFlash and yield gates, re-enable, yielding to company.
const std = @import("std");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const testing = std.testing;
const Generator = @import("../generate/generator.zig").Generator;

/// Number of attempts before the runtime gate considers disabling.
/// Below this we trust the prompt-time gate. Lower values make the gate trip sooner,
/// reducing regression-tail damage at the cost of fewer chances for slow-warmup
/// workloads to amortize spec overhead.
pub const RUNTIME_GATE_WARMUP: u64 = 5;

/// Minimum per-draft acceptance probability. Below this after warmup,
/// speculation is disabled for the rest of the request.
///
/// History: pre-v5 this gate compared `accepted/attempted` (per-round
/// average) against 0.30 — but with `block_size=4` the max value of that
/// ratio is 3.0, so the 0.30 threshold corresponded to ~10% per-draft
/// probability, well below where verify+draft overhead actually breaks
/// even. Empirically creative-content workloads regress at 22-47% per-draft
/// acceptance
/// while the gate stayed off (per-round avg 0.66-1.58, all above 0.30).
/// Switching to a per-draft probability with threshold 0.50 cleanly cuts
/// off the regressing tail while leaving heavy-echo workloads (84-97%
/// per-draft) running unmolested.
pub const RUNTIME_GATE_MIN_PER_DRAFT_RATE: f32 = 0.50;

/// Pure helper: should the runtime gate disable speculation given the
/// observed per-request stats? `drafts_per_round` is the number of
/// drafted tokens proposed in each verify (= `block_size - 1` for the
/// drafter, or `pld_draft_len` for PLD); we divide accepts by attempts ×
/// drafts_per_round to get the per-draft acceptance probability.
/// Returns true iff `attempted >= warmup` AND per-draft probability is
/// below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`.
///
/// `drafts_per_round == 0` is treated as "no speculative work happens
/// per round" → never trip (defensive — current callers always pass
/// >= 1).
pub fn runtimeGateShouldDisable(attempted: u64, accepted: u64, drafts_per_round: u32) bool {
    if (attempted < RUNTIME_GATE_WARMUP) return false;
    if (drafts_per_round == 0) return false;
    const drafts_proposed = attempted * @as(u64, drafts_per_round);
    const rate = @as(f32, @floatFromInt(accepted)) /
        @as(f32, @floatFromInt(drafts_proposed));
    return rate < RUNTIME_GATE_MIN_PER_DRAFT_RATE;
}

// DFlash has different economics from sequential-token drafters: one
// assistant call proposes the whole block and the M5 NAX verify lane makes
// wide blocks cheap. Per-draft percentage therefore penalizes block 16
// even when it is decisively profitable. The stable cross-width signal is
// accepted drafts per verify round. M5 Muse sweeps put code/tool traffic at
// 4.4-15.0 accepted/round and the regressing prose/vision class at
// 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
// workloads accepting 62-86% remain above it as well.
// Warmup 8, not 3: three rounds tripped on ONE dry opening
// (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
// tok/s; the same prompt restored from cache decoded at 73).
pub const DFLASH_GATE_WARMUP: u64 = 8;
pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
/// Absolute floor for a SPARSE target, applied after the width scaling.
/// See `scheduler.dflashGateMinimum` for the measurement it comes from —
/// a MoE verify reads every expert its block's positions route to, so its
/// break-even acceptance is several times a dense trunk's.
pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

/// The round policy (`dflash_policy.zig`) drives DFlash where the model declares it (`Caps.dflash_policy`); every
/// other model keeps the fixed block until it is A/B'd on its own.
pub fn dflashPolicyFor(config: *const model_mod.ModelConfig) bool {
    return config.caps().dflash_policy;
}

pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, min_avg: f32) bool {
    if (attempted < DFLASH_GATE_WARMUP) return false;
    const avg = @as(f32, @floatFromInt(accepted)) /
        @as(f32, @floatFromInt(attempted));
    return avg < min_avg;
}

// ── PLD yield gate (cold-path economics) ──
// The per-draft gate above only counts verify ROUNDS, so a workload where
// the n-gram lookup rarely matches never accumulates enough "attempts" to
// trip it — yet every no-match step pays PLD's unpipelined cold forward
// (measured −14% vs the async-pipelined `next()` on creative content).
// The yield gate instead counts EVERY enabled-mode nextPld step: if the
// speculation is yielding fewer than YIELD_GATE_MIN_YIELD extra (drafted,
// accepted) tokens per step after YIELD_GATE_WARMUP steps, the cold-path
// tax outweighs the wins → disable. Paired with `specShouldReenable`,
// which flips PLD back on when the generated tail turns repetitive.
// Warmup 32 (not higher): the re-enable check bounds the cost of a
// premature trip to ≤SPEC_REENABLE_INTERVAL pipelined-fallback steps,
// so we can gate early and recover the pipeline sooner on novel content.
/// Steps of enabled-mode PLD before the yield gate may trip. Every warmup step pays PLD's UNPIPELINED cold
/// forward (plus a synchronous host read of the sampled token) against `next()`'s async-pipelined step, so the
/// tax is a fraction of the AR step cost; 32 was calibrated before the Laguna mscale fix made that step ~3x
/// cheaper. See the `yield-gate warmup is 8` test for the sweep.
pub const YIELD_GATE_WARMUP: u64 = 8;
pub const YIELD_GATE_MIN_YIELD: f32 = 0.25;


pub fn yieldGateShouldDisable(steps_total: u64, accepted: u64) bool {
    if (steps_total < YIELD_GATE_WARMUP) return false;
    const yield_rate = @as(f32, @floatFromInt(accepted)) /
        @as(f32, @floatFromInt(steps_total));
    return yield_rate < YIELD_GATE_MIN_YIELD;
}

// ── Mid-request spec re-enable ──
// While the yield gate has PLD disabled, the COMMITTED sequence (prompt +
// generated) is re-scored every SPEC_REENABLE_INTERVAL steps: what
// fraction of the recent generated positions would have had a PLD lookup
// hit (their key-gram appears earlier in committed)? This catches the
// echo workload where the model repeats PROMPT content (file edits, tool
// results) — self-repetition scoring misses it because the echoed tail
// never repeats itself. Above the threshold, PLD is worth re-engaging at
// the cost of one pipeline drain.
pub const SPEC_REENABLE_INTERVAL: u64 = 32;
pub const SPEC_REENABLE_WINDOW: usize = 32;
pub const SPEC_REENABLE_MIN_FRACTION: f32 = 0.25;
pub const SPEC_REENABLE_MIN_TOKENS: usize = 16;

pub fn specShouldReenable(committed: []const u32, generated_len: usize) bool {
    if (generated_len < SPEC_REENABLE_MIN_TOKENS) return false;
    const window = @min(SPEC_REENABLE_WINDOW, generated_len);
    const frac = pld_index.tailMatchFraction(committed, window, 3);
    return frac >= SPEC_REENABLE_MIN_FRACTION;
}

// ── PLD cost gate ──
// The gates above count drafts; none of them knows what a round costs. Past a few thousand keys a 16-row verify
// (and the re-forward a partial accept owes) costs more wall time than the tokens it emits, at any acceptance.

/// Enabled steps (verify rounds and cold steps) before the cost gate asks what a plain step costs, so a request of
/// a few steps never probes, and before a loss within the margin's reach is believed.
pub const PLD_COST_PROBE_AFTER: u32 = 3;
pub const PLD_COST_WARMUP: u32 = 8;
/// A loss this large is believed as soon as both costs are known: at 65k keys a round's token costs 3.5 plain
/// ones, and every further round is a quarter second.
pub const PLD_COST_DECISIVE: f32 = 2.0;
/// Plain steps a probe runs; the first follows the hand-off from a round and is not a sample.
pub const PLD_COST_PROBE: u32 = 3;
/// PLD is dropped when its ms per emitted token passes a plain step's by this factor. Wide: a request's first
/// rounds are its dearest (few matches yet), and at 2k keys a 1.05 margin dropped a PLD that ends at half the
/// plain cost.
pub const PLD_COST_MARGIN: f32 = 1.25;
const PLD_COST_BETA: f32 = 0.25;

/// One request's measured price of PLD against plain decode. Pure: the generator feeds it wall times.
pub const PldCost = struct {
    /// EMAs over enabled steps: wall ms, and tokens emitted.
    spec_ms: f32 = 0,
    spec_tok: f32 = 0,
    spec_n: u32 = 0,
    /// EMA of a plain step's ms; `plain_n` 0 = never timed.
    plain_ms: f32 = 0,
    plain_n: u32 = 0,
    probe_left: u32 = 0,
    probed: bool = false,
    /// The next plain step follows a spec step: it pays the hand-off and is not a sample.
    hand_off: bool = false,
    /// The last spec stretch cost more per token than plain: no re-enable until `restart`.
    lost: bool = false,

    pub const Verdict = enum { keep, probe, disable };

    fn fold(ema: *f32, n: u32, x: f32) void {
        ema.* = if (n == 0) x else ema.* + PLD_COST_BETA * (x - ema.*);
    }

    /// An enabled step took `ms` and emitted `tokens`.
    pub fn spec(c: *PldCost, ms: f32, tokens: u32) Verdict {
        fold(&c.spec_ms, c.spec_n, ms);
        fold(&c.spec_tok, c.spec_n, @floatFromInt(tokens));
        c.spec_n += 1;
        if (c.spec_n < PLD_COST_PROBE_AFTER) return .keep;
        if (c.plain_n == 0) return if (c.probed) .keep else .probe;
        if (c.ratio() <= (if (c.spec_n < PLD_COST_WARMUP) PLD_COST_DECISIVE else PLD_COST_MARGIN)) return .keep;
        c.lost = true;
        return .disable;
    }

    /// A plain step took `ms`. True when it was a probe's last step.
    pub fn plain(c: *PldCost, ms: f32) bool {
        if (c.hand_off) {
            c.hand_off = false;
        } else {
            fold(&c.plain_ms, c.plain_n, ms);
            c.plain_n += 1;
        }
        if (c.probe_left == 0) return false;
        c.probe_left -= 1;
        return c.probe_left == 0;
    }

    /// The next `PLD_COST_PROBE` steps run plain, once a request.
    pub fn beginProbe(c: *PldCost) void {
        c.probed = true;
        c.probe_left = PLD_COST_PROBE;
        c.hand_off = true;
    }

    /// PLD stopped for another reason: the plain steps that follow are samples after the hand-off.
    pub fn handOff(c: *PldCost) void {
        c.hand_off = true;
    }

    /// PLD's ms per token over a plain step's; 0 while either is unmeasured.
    pub fn ratio(c: *const PldCost) f32 {
        if (c.spec_n == 0 or c.plain_n == 0 or c.spec_tok <= 0 or c.plain_ms <= 0) return 0;
        return c.spec_ms / c.spec_tok / c.plain_ms;
    }

    /// Whether PLD may run: nothing measured yet says its tokens cost more than plain ones.
    pub fn pays(c: *const PldCost) bool {
        return !c.lost and c.ratio() <= (if (c.spec_n < PLD_COST_WARMUP) PLD_COST_DECISIVE else PLD_COST_MARGIN);
    }

    /// A fresh spec stretch (the generated tail turned repetitive): it is judged on its own steps.
    pub fn restart(c: *PldCost) void {
        c.spec_n = 0;
        c.lost = false;
    }

    pub fn msPerToken(c: *const PldCost) f32 {
        return if (c.spec_tok > 0) c.spec_ms / c.spec_tok else 0;
    }
};

/// A lost context lets every this-many-th request speculate again, so a verdict cannot outlive what caused it.
pub const PLD_MEMORY_RETRY: u32 = 8;

/// What the cost gate last decided per context size on one loaded model, in memory only: a request in a context
/// that lost by `PLD_COST_DECISIVE` starts with PLD off instead of paying three rounds and a probe to find out
/// again.
pub const PldMemory = struct {
    /// One cell per power of two of the context's tokens.
    lost: [32]bool = @splat(false),
    skipped: [32]u32 = @splat(0),

    fn bucket(tokens: usize) usize {
        return std.math.log2_int(usize, @max(tokens, 1));
    }

    /// Asked once at a request's first PLD step: whether it starts with PLD off.
    pub fn startsOff(m: *PldMemory, tokens: usize) bool {
        const b = bucket(tokens);
        if (!m.lost[b]) return false;
        m.skipped[b] += 1;
        return m.skipped[b] % PLD_MEMORY_RETRY != 0;
    }

    /// A request's verdict at `tokens` of context: `ratio` is PLD's ms per token over plain's when it was dropped,
    /// or anything under the margin when it was kept.
    pub fn record(m: *PldMemory, tokens: usize, ratio: f32) void {
        const b = bucket(tokens);
        m.lost[b] = ratio >= PLD_COST_DECISIVE;
        if (!m.lost[b]) m.skipped[b] = 0;
    }
};

/// Decode plain in the batched group. Sticky for DFlash (plain rounds do not extend the
/// assistant context); PLD's periodic re-enable check resumes once the slot is solo.
pub fn yieldSpecToCompany(self: *Generator, mode: []const u8) void {
    if (self.spec_disabled_runtime) return;
    log.info("  {s}=disabled (company: decoding plain in the batched group)\n", .{mode});
    self.spec_disabled_runtime = true;
    self.spec_disable_reason = .company;
}

/// DFlash runtime economics gate. Sticky within the request: once a
/// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
/// less than its width-normalized request-class threshold, subsequent
/// ticks use the regular pipelined decoder through `nextDflash`'s entry
/// fallback.
pub fn checkDflashRuntimeGate(self: *Generator) void {
    if (self.spec_disabled_runtime or self.dflash_pol != null) return;
    if (!dflashGateShouldDisable(
        self.dflash_attempted,
        self.dflash_accepted_tokens,
        self.dflash_min_accepted_per_round,
    )) return;
    // The table's own measure outranks the calibrated bar: a round emitting
    // tokens cheaper than the plain step it would yield to stays on.
    if (self.xfm.round_cost.roundBeatsSerial(self.dflash_block_size -| 1, self.mtpKvLen()) == true) return;
    const avg = @as(f32, @floatFromInt(self.dflash_accepted_tokens)) /
        @as(f32, @floatFromInt(self.dflash_attempted));
    log.info(
        "  dflash=disabled (runtime yield {d:.2} accepted/round < {d:.2} after {d} attempts)\n",
        .{ avg, self.dflash_min_accepted_per_round, self.dflash_attempted },
    );
    self.spec_disabled_runtime = true;
}

/// Runtime acceptance gate for the drafter: after warmup, if the per-draft
/// acceptance probability is below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`,
/// disable speculation for the rest of this request. Sticky for the rest
/// of the generation.
pub fn checkDrafterRuntimeGate(self: *Generator) void {
    if (self.spec_disabled_runtime) return;
    const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
    if (!runtimeGateShouldDisable(self.drafter_attempted, self.drafter_accepted_tokens, drafts_per_round)) return;
    const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
    const rate: f32 = if (drafts_proposed > 0)
        @as(f32, @floatFromInt(self.drafter_accepted_tokens)) /
            @as(f32, @floatFromInt(drafts_proposed))
    else
        0.0;
    log.info(
        "  drafter=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
        .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.drafter_attempted },
    );
    self.spec_disabled_runtime = true;
}

// ── tests ──


test "Generator.runtimeGateShouldDisable below warmup never trips" {
    // Even with zero accepts, before the warmup count we trust the prompt-time
    // gate and never disable speculation mid-decode. drafts_per_round is the
    // typical drafter setting (block_size=4 → 3 drafts per round).
    try testing.expect(!Generator.runtimeGateShouldDisable(0, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(1, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP - 1, 0, 3));
}

test "Generator.runtimeGateShouldDisable trips at warmup with low per-draft rate" {
    // Synthetic low-accept scenario: 5 verify attempts, drafts_per_round=3
    // (drafter at block_size=4). 5 attempts × 3 = 15 drafts proposed.
    // 0 accepted → 0.00 < 0.50 → trip. Same with 1 accepted (0.067).
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 0, 3));
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 1, 3));
    // 7 accepted out of 15 = 0.467 — still below 0.50 → trip.
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 7, 3));
    // 8 accepted out of 15 = 0.533 → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 8, 3));
}

test "Generator.runtimeGateShouldDisable does not trip with high per-draft rate" {
    // Echo workloads on Gemma drafter: ~93% per-draft acceptance (E4B from
    // bench: 67/(24*3) = 93.1%). Well above threshold → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(24, 67, 3));
    // PLD heavy-echo: ~4 of 5 drafts accepted per attempt = 0.80 per-draft.
    try testing.expect(!Generator.runtimeGateShouldDisable(20, 80, 5));
    // Edge case at exactly the threshold (rate == 0.50) — strict less-than,
    // so does NOT trip.
    try testing.expect(!Generator.runtimeGateShouldDisable(10, 15, 3)); // 15/30 = 0.50
}

test "Generator.runtimeGateShouldDisable creative-content regression scenario" {
    // The Phase 1 bench's exact regression cases on creative prompts:
    //   E4B drafter (bs=4 → drafts_per_round=3): 39/59 attempts → 22.0% per-draft → trip
    //   E2B drafter (bs=2 → drafts_per_round=1): 31/66 attempts → 47.0% per-draft → trip
    //   31B drafter (bs=8 → drafts_per_round=7): 60/(38*7) → 22.6% per-draft → trip
    try testing.expect(Generator.runtimeGateShouldDisable(59, 39, 3)); // E4B creative
    try testing.expect(Generator.runtimeGateShouldDisable(66, 31, 1)); // E2B creative
    try testing.expect(Generator.runtimeGateShouldDisable(38, 60, 7)); // 31B creative
    // The 26B-A4B@bs=2 creative case: 37/(60*1) = 61.7% → above threshold,
    // so the runtime gate alone does NOT save it. MoE regressions need the
    // separate `default_enable_drafter` opt-out at startup.
    try testing.expect(!Generator.runtimeGateShouldDisable(60, 37, 1));
}

test "Generator.runtimeGateShouldDisable handles drafts_per_round=0" {
    // Defensive: if a caller somehow passes a degenerate config (block_size=1
    // → drafts_per_round=0), don't divide by zero. We return false (no trip).
    try testing.expect(!Generator.runtimeGateShouldDisable(100, 0, 0));
}

test "Generator.dflashGateShouldDisable uses accepted yield across block widths" {
    // Never decide from fewer than the DFlash warmup number of paid verifies.
    try testing.expect(!Generator.dflashGateShouldDisable(
        Generator.DFLASH_GATE_WARMUP - 1,
        0,
        Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    ));

    // Muse prose/vision measured 1.0-1.5 accepted drafts per round at both
    // block 8 and block 16: that class loses to serial and must fall back.
    const w = Generator.DFLASH_GATE_WARMUP;
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 2 - 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Exactly two is the strict break-even boundary; code/tool traffic at
    // 4.4+ and echo traffic near a full block remain on DFlash.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 2, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 5, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 15, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Thinking preambles recovered from ~1.4 early to 4.4 whole-request; the
    // resolved-mode threshold keeps that path alive while still cutting off
    // a truly non-yielding reasoning request.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w + w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w - w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
}

test "Generator.yieldGateShouldDisable trips on cold-path-dominated workloads" {
    // The baseline regression: PLD forced on for a creative essay
    // prompt where the n-gram lookup almost never matches. The per-draft gate
    // never trips (it only counts verify ROUNDS, and there are few), but every
    // step pays the unpipelined cold forward → −14% measured on E2B. The
    // yield gate counts ALL enabled-mode steps: accepted-drafted-tokens per
    // step below the threshold after warmup → disable.
    // Creative: 128 steps, ~6 drafted tokens accepted → yield 0.047 → trip.
    try testing.expect(Generator.yieldGateShouldDisable(128, 6));
    // Heavy echo: 40 steps, 80 accepted (2.0/step) → stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(40, 80));
    // Inside warmup: never trip, even at zero yield.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP - 1, 0));
    // Exactly at warmup with healthy yield: stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP, Generator.YIELD_GATE_WARMUP));
}

test "yield-gate warmup is 8 — the cold-path tax tripled when the AR step got 3x cheaper" {
    // This constant is pure economics and the economics MOVED. Every warmup
    // step pays PLD's unpipelined cold forward plus a synchronous host read of
    // the sampled token, against `next()`'s async-pipelined step — an
    // ~absolute tax measured as a share of the AR step. Fixing the Laguna
    // mscale promotion made that AR step ~3x cheaper, so the same tax became a
    // ~3x larger share and 32 steps of it stopped being affordable.
    //
    // Swept on Laguna XS (one boot, serial vs unconstrained alternating per
    // request, server-reported timings, 5 runs median), against a matrix that
    // deliberately contains PLD's WIN case as well as its loss cases —
    // tuning on loss cases alone drives the warmup to zero and throws the win
    // away:
    //
    //   warmup   echo-edit   code-edit  free-form   explain      qa
    //       32     +76.9%       -4.3%      -1.2%     -1.4%     -7.2%
    //       16     +70.1%       -3.4%      -0.4%     -0.3%     -0.8%
    //        8     +77.4%       +0.3%      -0.2%     -0.3%     -1.5%
    //        4     +76.6%       +3.9%      -0.2%     -0.1%     -1.7%
    //
    // 8 recovers essentially the whole loss while leaving the +77% untouched.
    // 4 measured no worse, but a gate that decides on four observations is
    // fitting noise, and a short preamble before a file echo is the NORMAL
    // agent shape — exactly what a too-eager trip would punish. A premature
    // trip is bounded anyway: `specShouldReenable` re-checks every
    // SPEC_REENABLE_INTERVAL steps.
    try testing.expectEqual(@as(u64, 8), Generator.YIELD_GATE_WARMUP);

    // Sweepable without a rebuild, same contract as `runtimeGateWarmup`:
    // out-of-range values fall back rather than silently disabling the gate.
    try testing.expectEqual(Generator.YIELD_GATE_WARMUP, YIELD_GATE_WARMUP);
}

test "Generator.specShouldReenable gates mid-request PLD re-activation" {
    // Disabled-mode periodic check on the COMMITTED sequence (prompt +
    // generated). The decisive case: the model echoes PROMPT content (file
    // edit / tool result) after a novel preamble tripped the yield gate. The
    // echoed tail never repeats ITSELF, so self-repetition scoring misses it;
    // tailMatchFraction sees the prompt occurrence.
    var committed: [96]u32 = undefined;
    // prompt = 48-token "file", generated = 16 novel preamble + 32 echo of the file
    for (committed[0..48], 0..) |*t, i| t.* = @intCast(i + 100);
    for (committed[48..64], 0..) |*t, i| t.* = @intCast(i + 9000);
    for (committed[64..96], 0..) |*t, i| t.* = @intCast(i + 100);
    try testing.expect(Generator.specShouldReenable(&committed, 48));

    // Fully novel committed sequence → stay disabled.
    var novel: [96]u32 = undefined;
    for (&novel, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    try testing.expect(!Generator.specShouldReenable(&novel, 48));

    // Too little generated yet → not enough signal, stay disabled.
    try testing.expect(!Generator.specShouldReenable(&committed, 8));
}

test "PldCost: a few steps never probe; then a probe, and a loss of twice the plain cost ends PLD at once" {
    var c: PldCost = .{};
    for (0..PLD_COST_PROBE_AFTER - 1) |_| try std.testing.expectEqual(PldCost.Verdict.keep, c.spec(270, 3));
    // Nothing says yet what a plain token costs.
    try std.testing.expectEqual(PldCost.Verdict.probe, c.spec(270, 3));
    c.beginProbe();
    // The hand-off step is not a sample; the probe ends on its last counted step.
    try std.testing.expect(!c.plain(90));
    try std.testing.expect(!c.plain(27));
    try std.testing.expect(c.plain(27));
    // 90 ms a token in rounds against 27 plain.
    try std.testing.expect(!c.pays());
    try std.testing.expectEqual(PldCost.Verdict.disable, c.spec(270, 3));
    try std.testing.expect(c.lost);
}

test "PldCost: a loss inside twice the plain cost waits for the warmup, and the probe runs once" {
    var c: PldCost = .{};
    for (0..PLD_COST_PROBE_AFTER) |_| _ = c.spec(36, 1);
    c.beginProbe();
    for (0..PLD_COST_PROBE) |_| _ = c.plain(27);
    // 36 against 27: early rounds are the dearest, so not yet.
    try std.testing.expect(c.pays());
    for (PLD_COST_PROBE_AFTER..PLD_COST_WARMUP - 1) |_| try std.testing.expectEqual(PldCost.Verdict.keep, c.spec(36, 1));
    try std.testing.expectEqual(PldCost.Verdict.disable, c.spec(36, 1));

    // A probe that timed nothing (the request's plain steps were all hand-offs) is not repeated.
    var blind: PldCost = .{ .probed = true };
    for (0..PLD_COST_WARMUP + 4) |_| try std.testing.expectEqual(PldCost.Verdict.keep, blind.spec(500, 1));
}

test "PldCost: rounds cheaper per token than a plain step keep PLD on, within the margin too" {
    var c: PldCost = .{};
    for (0..PLD_COST_WARMUP) |_| _ = c.spec(40, 4);
    c.beginProbe();
    for (0..PLD_COST_PROBE) |_| _ = c.plain(22);
    try std.testing.expect(c.pays());
    try std.testing.expectEqual(PldCost.Verdict.keep, c.spec(40, 4));
    // 26 ms a token against 22 plain is inside the margin.
    var near: PldCost = .{};
    for (0..PLD_COST_WARMUP) |_| _ = near.spec(26, 1);
    near.beginProbe();
    for (0..PLD_COST_PROBE) |_| _ = near.plain(22);
    try std.testing.expectEqual(PldCost.Verdict.keep, near.spec(26, 1));
}

test "PldCost: cold steps count as PLD's cost, and a later stretch is judged on its own" {
    var c: PldCost = .{};
    c.beginProbe();
    for (0..PLD_COST_PROBE) |_| _ = c.plain(20);
    // One token a step at 30 ms (no match, an unpipelined forward): slower than plain.
    var v = PldCost.Verdict.keep;
    for (0..PLD_COST_WARMUP) |_| v = c.spec(30, 1);
    try std.testing.expectEqual(PldCost.Verdict.disable, v);
    try std.testing.expect(!c.pays());
    // The tail turns repetitive and PLD is given a fresh stretch.
    c.restart();
    try std.testing.expect(c.pays());
    for (0..PLD_COST_WARMUP) |_| v = c.spec(60, 6);
    try std.testing.expectEqual(PldCost.Verdict.keep, v);
}

test "PldMemory: a decisive loss turns PLD off for that context size, with a retry, until a request keeps it" {
    var m: PldMemory = .{};
    try std.testing.expect(!m.startsOff(60_000));
    m.record(60_000, 3.4);
    // 36k and 65k share a power of two; 2k does not.
    for (1..PLD_MEMORY_RETRY) |_| try std.testing.expect(m.startsOff(36_000));
    try std.testing.expect(!m.startsOff(65_000));
    try std.testing.expect(m.startsOff(65_000));
    try std.testing.expect(!m.startsOff(2_200));
    // A loss inside twice the plain cost is not remembered, and a kept request clears the cell.
    m.record(2_200, 1.4);
    try std.testing.expect(!m.startsOff(2_200));
    m.record(60_000, 0.5);
    try std.testing.expect(!m.startsOff(36_000));
}
