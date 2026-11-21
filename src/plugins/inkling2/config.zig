//! Inkling 2's claim and parse (`inkling2_loop`, mlx-lm's layout): `text_config`, the loop (`loop_config`: layers
//! `core_start..core_end` run `num_passes` times) and the cache layout, one entry per layer VISIT (each visit keeps
//! its own KV and its four short-conv windows). MLX-free. A config the forward does not model is refused by name.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const llama = @import("../llama/config.zig");

const field = llama.field;
const int = llama.int;
const float = llama.float;

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "inkling2",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{"inkling2_loop"};
pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 128;
pub const max_visits = 512;

/// One attention geometry (the global layers', or the sliding layers').
pub const Heads = struct { heads: u32, kv_heads: u32, head_dim: u32 };

pub const Config = struct {
    vocab_size: u32,
    /// The logits the model emits: the padded vocab's real rows.
    unpadded_vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    full: Heads,
    sliding: Heads,
    sliding_window: u32,
    global: [max_layers]bool = @splat(false),
    d_rel: u32,
    rel_extent: u32,
    log_n_floor: u32,
    log_alpha: f32 = 0.1,
    rms_norm_eps: f32 = 1e-6,
    qk_clip: f32,
    rel_clip: f32,
    sconv_kernel: u32 = 4,
    /// Layers per block of the attention-residual bank (a block opens at every multiple).
    block_layers: u32,
    num_experts: u32,
    top_k: u32,
    route_scale: f32 = 1.0,
    gate_softcap: f32,
    up_softcap: f32,
    logit_softcap: f32,
    /// The looped layers `[core_start, core_end)` and how often they run; `passes` 1 with an empty core = no loop.
    core_start: u32 = 0,
    core_end: u32 = 0,
    passes: u32 = 1,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",

    pub fn core(c: *const Config) u32 {
        return c.core_end - c.core_start;
    }

    /// Layer applications per token: the cache entries of a slot.
    pub fn visits(c: *const Config) u32 {
        return c.num_hidden_layers + c.core() * (c.passes - 1);
    }

    /// The layer visit `v` applies.
    pub fn layerOf(c: *const Config, v: u32) u32 {
        if (v < c.core_start) return v;
        const looped = c.core() * c.passes;
        if (v < c.core_start + looped) return c.core_start + (v - c.core_start) % c.core();
        return v - looped + c.core();
    }

    pub fn heads(c: *const Config, layer: u32) Heads {
        return if (c.global[layer]) c.full else c.sliding;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "inkling2_loop: " ++ why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "inkling2_loop: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "inkling2_loop: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, model_types[0])) return unsupported("another model_type");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    const tc = field(root, "text_config") orelse return unsupported("no text_config");
    if (tc != .object) return unsupported("text_config");
    const cfg = tc.object;

    var c: Config = .{
        .vocab_size = int(field(cfg, "vocab_size")) orelse return unsupported("vocab_size"),
        .unpadded_vocab_size = int(field(cfg, "unpadded_vocab_size")) orelse return unsupported("unpadded_vocab_size"),
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .intermediate_size = int(field(cfg, "intermediate_size")) orelse return unsupported("intermediate_size"),
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .full = .{
            .heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
            .kv_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
            .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
        },
        .sliding = .{
            .heads = int(field(cfg, "swa_num_attention_heads")) orelse return unsupported("swa_num_attention_heads"),
            .kv_heads = int(field(cfg, "swa_num_key_value_heads")) orelse return unsupported("swa_num_key_value_heads"),
            .head_dim = int(field(cfg, "swa_head_dim")) orelse return unsupported("swa_head_dim"),
        },
        .sliding_window = int(field(cfg, "sliding_window_size")) orelse return unsupported("sliding_window_size"),
        .d_rel = int(field(cfg, "d_rel")) orelse return unsupported("d_rel"),
        .rel_extent = int(field(cfg, "rel_extent")) orelse return unsupported("rel_extent"),
        .log_n_floor = int(field(cfg, "log_scaling_n_floor")) orelse return unsupported("log_scaling_n_floor"),
        .qk_clip = @floatCast(float(field(cfg, "qk_l2_clip_threshold") orelse return unsupported("no qk_l2_clip_threshold")) orelse return unsupported("qk_l2_clip_threshold")),
        .rel_clip = @floatCast(float(field(cfg, "rel_l2_clip_threshold") orelse return unsupported("no rel_l2_clip_threshold")) orelse return unsupported("rel_l2_clip_threshold")),
        .block_layers = int(field(cfg, "n_layers_per_attn_res_block")) orelse return unsupported("n_layers_per_attn_res_block"),
        .num_experts = int(field(cfg, "n_routed_experts")) orelse return unsupported("n_routed_experts"),
        .top_k = int(field(cfg, "num_experts_per_tok")) orelse return unsupported("num_experts_per_tok"),
        .gate_softcap = @floatCast(float(field(cfg, "mlp_gate_softcap") orelse return unsupported("no mlp_gate_softcap")) orelse return unsupported("mlp_gate_softcap")),
        .up_softcap = @floatCast(float(field(cfg, "mlp_up_softcap") orelse return unsupported("no mlp_up_softcap")) orelse return unsupported("mlp_up_softcap")),
        .logit_softcap = @floatCast(float(field(cfg, "final_logit_softcapping") orelse return unsupported("no final_logit_softcapping")) orelse return unsupported("final_logit_softcapping")),
    };
    if (field(cfg, "sconv_kernel_size")) |v| c.sconv_kernel = int(v) orelse return unsupported("sconv_kernel_size");
    inline for (.{ .{ "rms_norm_eps", &c.rms_norm_eps }, .{ "route_scale", &c.route_scale }, .{ "log_scaling_alpha", &c.log_alpha } }) |f| {
        if (field(cfg, f[0])) |v| f[1].* = @floatCast(float(v) orelse return unsupported(f[0]));
    }
    // What the forward runs unconditionally.
    inline for (.{ .{ "use_value_norm_no_gamma", true }, .{ "attn_res_normalize_values", true }, .{ "q_bias", false }, .{ "o_bias", false } }) |f| {
        if (field(cfg, f[0])) |v| if (v != .bool or v.bool != f[1]) return unsupported(f[0] ++ " other than the released value");
    }
    if (field(cfg, "n_shared_experts")) |v| if ((int(v) orelse return unsupported("n_shared_experts")) != 0) return unsupported("shared experts");
    if (field(cfg, "loop_config")) |lv| if (lv != .null) {
        if (lv != .object) return unsupported("loop_config");
        const loop = lv.object;
        if (field(loop, "is_varloop")) |v| if (v != .bool or v.bool) return unsupported("a variable loop");
        c.core_start = int(field(loop, "core_start")) orelse return unsupported("core_start");
        c.core_end = int(field(loop, "core_end")) orelse return unsupported("core_end");
        c.passes = int(field(loop, "num_passes")) orelse return unsupported("num_passes");
    };
    const ok_heads = struct {
        fn f(h: Heads) bool {
            return h.heads > 0 and h.kv_heads > 0 and h.heads % h.kv_heads == 0 and h.head_dim > 0 and h.head_dim <= 512;
        }
    }.f;
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.unpadded_vocab_size == 0 or c.unpadded_vocab_size > c.vocab_size or
        c.hidden_size == 0 or c.hidden_size > 1 << 16 or c.intermediate_size == 0 or c.num_hidden_layers == 0 or
        c.num_hidden_layers > max_layers or !ok_heads(c.full) or !ok_heads(c.sliding) or c.sliding_window == 0 or
        c.d_rel == 0 or c.rel_extent == 0 or c.log_n_floor == 0 or c.sconv_kernel < 2 or c.sconv_kernel > 64 or
        c.block_layers == 0 or c.num_experts == 0 or c.num_experts > 4096 or c.top_k == 0 or c.top_k > c.num_experts or
        c.gate_softcap <= 0 or c.up_softcap <= 0 or c.logit_softcap <= 0 or c.qk_clip <= 0 or c.rel_clip <= 0 or
        c.passes == 0 or c.passes > 16 or c.core_start > c.core_end or c.core_end > c.num_hidden_layers or
        (c.passes > 1 and c.core() == 0))
        return unsupported("dimensions out of range");
    // The message bank is the looped layers' block states: the core opens and closes on a block boundary.
    if (c.passes > 1 and (c.core_start % c.block_layers != 0 or c.core_end % c.block_layers != 0)) return unsupported("a loop core off the block boundaries");
    if (c.visits() > max_visits) return unsupported("dimensions out of range");

    const local = field(cfg, "local_layer_ids") orelse return unsupported("no local_layer_ids");
    if (local != .array) return unsupported("local_layer_ids");
    @memset(c.global[0..c.num_hidden_layers], true);
    for (local.array.items) |item| {
        const i = int(item) orelse return unsupported("local_layer_ids");
        if (i < c.num_hidden_layers) c.global[i] = false;
    }

    var p: abi.Parsed = .{};
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.unpadded_vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        // The engine's per-layer state is per visit.
        .num_hidden_layers = c.visits(),
        .num_attention_heads = c.full.heads,
        .num_key_value_heads = c.full.kv_heads,
        .head_dim = c.full.head_dim,
        .num_experts_per_tok = c.top_k,
        .moe_intermediate_size = c.intermediate_size,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(model_types[0]),
    };
    llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    if (int(field(cfg, "model_max_length"))) |n| {
        s.max_position_embeddings = n;
        s.context_cap = n;
    }
    c.quant = llama.readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    p.cache.ssm_state = .{ .entries = c.visits(), .dtype = .f32 };
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "inkling2_loop: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "inkling2_loop: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// One entry per visit: a global layer's keeps full KV, a sliding one its window; no layer ropes.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.visits(), cap);
    for (out[0..n], 0..) |*o, v| {
        const layer = c.layerOf(@intCast(v));
        const g = c.global[layer];
        const h = c.heads(layer);
        o.* = .{
            .kind = if (g) .full else .sliding,
            .nope = true,
            .window = if (g) 0 else c.sliding_window,
            .kv_heads = h.kv_heads,
            .k_dim = h.head_dim,
            .v_dim = h.head_dim,
        };
    }
    return n;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        gpa.free(c.block_json);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

// ── tests ──

const testing = std.testing;

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

/// mlx-community/Inkling-2-Mini-4bit's config.json, trimmed to the keys the engine reads.
pub const mini_4bit =
    \\{"model_type":"inkling2_loop","eos_token_id":200006,"text_config":{"model_max_length":1048576,
    \\"hidden_size":2304,"num_hidden_layers":24,"vocab_size":201024,"num_attention_heads":24,"num_key_value_heads":8,
    \\"head_dim":128,"d_rel":16,"rel_extent":1024,"q_bias":false,"o_bias":false,"log_scaling_n_floor":8192,
    \\"log_scaling_alpha":0.1,"rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
    \\"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,"sconv_kernel_size":4,
    \\"unpadded_vocab_size":200058,"final_logit_softcapping":30.0,"swa_head_dim":128,"swa_num_attention_heads":24,
    \\"swa_num_key_value_heads":8,"sliding_window_size":1024,"n_layers_per_attn_res_block":6,
    \\"attn_res_normalize_values":true,"n_routed_experts":128,"num_experts_per_tok":8,"n_shared_experts":0,
    \\"intermediate_size":1152,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,"use_gate_bias":true,
    \\"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
    \\"loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}},
    \\"quantization":{"group_size":64,"bits":4,"mode":"affine"}}
;

test "inkling2 config: the Mini pack's loop gives 48 visits, each with its own cache entry of its layer's kind" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(mini_4bit, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 48), c.visits());
    try testing.expectEqual(@as(u32, 48), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 48), p.cache.ssm_state.entries);
    try testing.expectEqual(@as(u32, 200058), p.shell.vocab_size);
    // Prelude, three passes over 6..17, coda.
    const want = [_]u32{ 0, 5, 6, 17, 6, 17, 6, 17, 18, 23 };
    for ([_]u32{ 0, 5, 6, 17, 18, 29, 30, 41, 42, 47 }, want) |v, layer| try testing.expectEqual(layer, c.layerOf(v));
    var lay: [48]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 48), layout(&p, null, &lay, lay.len));
    // Layer 11 is global: visits 11, 23 and 35.
    for ([_]usize{ 5, 11, 23, 35, 41, 47 }) |v| try testing.expect(lay[v].kind == .full and lay[v].window == 0);
    for ([_]usize{ 0, 6, 18, 30, 42 }) |v| try testing.expect(lay[v].kind == .sliding and lay[v].window == 1024 and lay[v].nope);
}

test "inkling2 config: a variable loop, shared experts or a core off the block boundaries is refused unsupported" {
    inline for (.{ .{ "\"is_varloop\":false", "\"is_varloop\":true" }, .{ "\"n_shared_experts\":0", "\"n_shared_experts\":1" }, .{ "\"core_start\":6", "\"core_start\":5" }, .{ "\"use_value_norm_no_gamma\":true", "\"use_value_norm_no_gamma\":false" } }) |swap| {
        const json = comptime blk: {
            @setEvalBranchQuota(100_000);
            const at = std.mem.indexOf(u8, mini_4bit, swap[0]).?;
            break :blk mini_4bit[0..at] ++ swap[1] ++ mini_4bit[at + swap[0].len ..];
        };
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
