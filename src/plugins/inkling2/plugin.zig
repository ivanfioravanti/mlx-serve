//! Thinking Machines Inkling 2 (`inkling2_loop`, text), mlx-lm's layout. A looped trunk: the layers before the core
//! run once, the core `num_passes` times (each pass restarts from the prelude's output and hears the previous pass
//! only through message gates), then the tail. Every layer reads its attention and MLP inputs by depth attention
//! over a bank of block states; a layer opening a block replaces the residual with its attention output. Each layer
//! VISIT has its own KV and conv windows (`c.layer` is the visit). Images splice after the embedding norm
//! (`vision.zig`). Serial, spec off.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");
const loader = @import("../llama/load.zig");
const attention = @import("attention.zig");
const blocks = @import("blocks.zig");
const compiled = @import("compiled.zig");
const moe = @import("moe.zig");
const vision = @import("vision.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = ops.Array;
const Graph = ops.Graph;
const Linear = b.linear.Linear;
const gpa = std.heap.c_allocator;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = config.layout },
    .media = &.{ .tower = &vision.tower, .prepare = &vision.prepare },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &trunkWeight },
};

/// Every term is the engine's default arithmetic over the shell and the layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Layer = struct {
    global: bool,
    /// Opens a block: its normed input joins the bank and the residual restarts from its attention output.
    boundary: bool,
    attn: attention.Weights,
    /// The q, k, v input norms (a sliding layer has one, repeated).
    in_norm: [3]Array,
    /// `[1 or 3, hidden]` f32; null on layer 0.
    attn_query: Array = .{},
    /// `[1, hidden]` f32.
    mlp_query: Array,
    post_norm: Array,
    mlp: moe.Weights,
    attn_conv: Array,
    mlp_conv: Array,
    /// q, k, v, mlp; a looped layer's.
    gates: ?[4]blocks.MessageGate = null,
};

pub const Head = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = false;
    pub const Weights = struct { lin: Linear, vocab: c_int, softcap: f32 };

    /// The padded vocab's real rows, softcapped in f32.
    pub fn forward(w: *const Weights, c: *block.Ctx, x: Array) !Array {
        const g = c.g;
        const logits = try g.astype(try head_block.project(g, &w.lin, x), .float32);
        const capped = try g.mulScalar(try g.tanh(try g.mulScalar(logits, 1.0 / w.softcap)), w.softcap);
        return g.sliceAxis(capped, 2, 0, w.vocab);
    }
};

pub const Model = struct {
    pub const batches_decode = false;

    pub const Weights = struct {
        embed: b.embed.TableEmbed.Weights,
        embed_norm: Array,
        layers: []const Layer,
        /// `[core, 4, hidden]` f32: each looped layer's q, k, v and mlp message queries.
        message_query: Array = .{},
        final_query: Array,
        final_norm: Array,
        head: Head.Weights,
        core_start: u32,
        core_end: u32,
        passes: u32,
        eps: f32,
    };

    const Trunk = struct {
        w: *const Weights,
        c: *block.Ctx,
        g: *Graph,
        run: *const run_mod.Run,
        h: Array,
        /// A residual a later pass restarts from: not dropped when the stream moves on.
        pinned: Array = .{},
        bank: [64]Array = undefined,
        blocks: usize = 0,
        visit: u32 = 0,
        visits: u32,

        fn move(t: *Trunk, to: Array) !void {
            const old = t.h;
            t.h = try t.g.keep(to);
            if (old.ctx != t.pinned.ctx) t.g.drop(old);
        }

        /// Layer `li` on the residual; `messages` `[1, L, 4, hidden]` in a pass after the first.
        fn apply(t: *Trunk, li: usize, messages: ?Array) !void {
            const L = &t.w.layers[li];
            const c = t.c;
            var lg = t.g.scope();
            defer lg.deinit();
            const g = &lg;
            c.g = g;
            defer c.g = t.g;
            c.layer = t.visit;
            const eps = t.w.eps;
            const dt = ops.dtype(t.h);
            const hidden = ops.shape(t.h)[2];
            const kvd = L.attn.kv_heads * L.attn.head_dim;
            const normed_in = try blocks.rms(g, try blocks.f32Of(g, t.h), 1e-6);

            // The visit's conv windows: k | v | attention output | MLP output.
            const st = try kv.state(c, .conv);
            var prev: [4]?Array = @splat(null);
            if (st) |a| {
                var at: c_int = 0;
                for (&prev, [_]c_int{ kvd, kvd, hidden, hidden }) |*p, n| {
                    p.* = try g.sliceAxis(a, 2, at, at + n);
                    at += n;
                }
            }

            const attn_in: ?Array = if (L.attn_query.ctx != null) try blocks.depth(g, t.bank[0..t.blocks], normed_in, L.attn_query) else null;
            var qkv: [3]Array = @splat(t.h);
            if (attn_in) |x| {
                if (L.global) {
                    for (&qkv, 0..) |*s, j| s.* = try blocks.row(g, x, @intCast(j));
                } else qkv = @splat(try blocks.row(g, x, 0));
            }
            if (messages) |m| for (&qkv, 0..) |*s, j| {
                s.* = try L.gates.?[j].forward(g, s.*, try blocks.row(g, m, @intCast(j)));
            };
            var normed: [3]Array = undefined;
            for (&normed, qkv, 0..) |*n, s, j| {
                n.* = if (j > 0 and s.ctx == qkv[0].ctx and L.in_norm[j].ctx == L.in_norm[0].ctx) normed[0] else try g.astype(try g.rmsNorm(s, L.in_norm[j], eps), dt);
            }
            const a = try attention.forward(&L.attn, c, normed[0], normed[1], normed[2], prev[0], prev[1]);
            const ac = try blocks.sconv(g, a.out, L.attn_conv, prev[2]);

            var res: Array = undefined;
            if (L.boundary) {
                res = try blocks.f32Of(g, ac.out);
                t.bank[t.blocks] = try t.g.keep(normed_in);
                t.blocks += 1;
            } else res = try compiled.sum(g, t.h, ac.out);

            var x = try blocks.row(g, try blocks.depth(g, t.bank[0..t.blocks], try blocks.rms(g, res, 1e-6), L.mlp_query), 0);
            if (messages) |m| x = try L.gates.?[3].forward(g, x, try blocks.row(g, m, 3));
            const y = try moe.forward(&L.mlp, g, try g.astype(try g.rmsNorm(x, L.post_norm, eps), dt));
            const mc = try blocks.sconv(g, y, L.mlp_conv, prev[3]);
            try kv.setState(c, .conv, try g.concat(&.{ a.k.state, a.v.state, ac.state, mc.state }, 2));
            // A verify forward's rejected rows are rolled back from the conv inputs.
            if (c.capture_spec) try kv.setState(c, .conv_input, try g.concat(&.{ a.k.input, a.v.input, ac.input, mc.input }, 2));

            try t.move(try compiled.sumAs(g, res, mc.out));
            try t.run.capture(t.g, t.visit, t.h);
            try t.run.cadence(t.visit, t.visits, t.h);
            t.visit += 1;
        }
    };

    /// One forward of `ids` (`[1, rows]` int32). Results are owned by `g`.
    pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: run_mod.Run) !run_mod.Out {
        if (run.mode == .batched_rows) return error.BatchedRowsUnsupported;
        var c = run.ctx(g, shared);
        const embedded = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &c, ids), w.embed_norm, w.eps);
        const n: u32 = @intCast(w.layers.len);
        const core = w.core_end - w.core_start;
        var t: Trunk = .{ .w = w, .c = &c, .g = g, .run = &run, .h = try run.input(g, embedded, ids), .visits = n + core * (w.passes - 1) };

        for (0..w.core_start) |i| try t.apply(i, null);
        t.pinned = t.h;
        const base = t.blocks;
        var heard: [64]Array = undefined;
        for (0..w.passes) |pass| {
            if (pass == 0) {
                for (w.core_start..w.core_end) |i| try t.apply(i, null);
                continue;
            }
            // The previous pass as the next one hears it: its block states after the first, and its output.
            const said = t.blocks - (base + 1);
            @memcpy(heard[0..said], t.bank[base + 1 .. t.blocks]);
            const last = try g.keep(try blocks.rms(g, try blocks.f32Of(g, t.h), 1e-6));
            if (t.h.ctx != t.pinned.ctx) g.drop(t.h);
            t.h = t.pinned;
            t.blocks = base;
            for (w.core_start..w.core_end) |i| {
                const j: c_int = @intCast(i - w.core_start);
                const query = try g.reshape(try g.sliceAxis(w.message_query, 0, j, j + 1), ops.shape(w.message_query)[1..]);
                try t.apply(i, try blocks.depth(g, heard[0..said], last, query));
            }
        }
        t.pinned = .{};
        for (w.core_end..n) |i| try t.apply(i, null);

        const normed = try blocks.rms(g, try blocks.f32Of(g, t.h), 1e-6);
        const mixed = try blocks.row(g, try blocks.depth(g, t.bank[0..t.blocks], normed, w.final_query), 0);
        const final = try g.astype(try g.rmsNorm(mixed, w.final_norm, w.eps), ops.dtype(t.h));
        // A drafter's last tap: the final-normed output, published one past the last visit.
        try run.capture(g, t.visits, final);
        return run.finish(g, &c, final, Head, &w.head);
    }
};

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Layer = &.{},
    owned: loader.Owned = .{},
    /// Null on a pack shipped without its tower.
    tower: ?vision.Tower = null,

    fn release(m: *Loaded) void {
        if (m.tower) |*t| t.deinit();
        m.owned.release();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

/// The bare embedding table and LM head a drafter reads (no embedding norm, softcap or vocab trim).
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *@import("../../abi/quant.zig").Weight) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    b.trunk_weight.trunkWeight(switch (which) {
        .embed => &m.base.weights.embed.table,
        .head => &m.base.weights.head.lin,
    }, out);
    return .ok;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

/// MLX's NVFP4 global scale is the tensor's amax; its codes times their block scales span `6 * 448`.
const nvfp4_span: f32 = 6.0 * 448.0;

const Load = struct {
    r: *loader.Reader,
    o: *loader.Owned,
    nb: [256]u8 = undefined,

    fn name(l: *Load, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return l.r.name(&l.nb, fmt, args);
    }

    fn need(l: *Load, comptime fmt: []const u8, args: anytype) !Array {
        const a = try l.r.need(try l.name(fmt, args));
        try l.o.own(a);
        return a;
    }

    fn proj(l: *Load, in_dim: u32, comptime fmt: []const u8, args: anytype) !Linear {
        const p = try l.r.proj(try l.name(fmt, args), in_dim);
        try l.o.linear(p);
        return p;
    }

    /// An evaluated f32 copy of a tensor, reshaped (`sh` empty = as stored), transposed (`[out, in]` to `[in, out]`)
    /// or scaled.
    fn made(l: *Load, comptime fmt: []const u8, args: anytype, how: struct { sh: []const c_int = &.{}, t: bool = false, dt: ?mlx.mlx_dtype = .float32, mul: f32 = 1 }) !Array {
        const src = try l.r.need(try l.name(fmt, args));
        defer _ = mlx.mlx_array_free(src);
        var g = Graph.init(gpa, l.r.s);
        defer g.deinit();
        var x: Array = src;
        if (how.dt) |dt| x = try g.astype(x, dt);
        if (how.sh.len > 0) x = try g.reshape(x, how.sh);
        if (how.t) x = try g.transpose(x, &.{ 1, 0 });
        if (how.mul != 1) x = try g.mulScalar(x, how.mul);
        x = try g.contiguous(x);
        try ops.eval(x);
        const kept = g.escape(x);
        try l.o.own(kept);
        return kept;
    }

    fn gate(l: *Load, i: usize, comptime which: []const u8) !blocks.MessageGate {
        return .{
            .down = try l.made("layers.{d}." ++ which ++ "_message_gate.down_proj.weight", .{i}, .{ .t = true, .dt = null }),
            .forget = try l.made("layers.{d}." ++ which ++ "_message_gate.forget_gate_proj.weight", .{i}, .{ .t = true }),
            .message = try l.made("layers.{d}." ++ which ++ "_message_gate.message_gate_proj.weight", .{i}, .{ .t = true }),
        };
    }

    /// An expert bank and, where the pack carries one, each expert's NVFP4 global scale as the multiplier on its output.
    fn bank(l: *Load, i: usize, comptime which: []const u8, in_dim: u32, scale: *Array) !b.moe.Bank {
        const bk = try l.r.bank(try l.name("layers.{d}.mlp.experts." ++ which, .{i}), in_dim);
        for ([_]Array{ bk.w, bk.s, bk.b }) |a| try l.o.own(a);
        if (!l.r.has(try l.name("layers.{d}.mlp.experts." ++ which ++ ".global_scale", .{i}))) return bk;
        if (bk.qp.mode != .nvfp4) return error.GlobalScaleOffNvfp4;
        scale.* = try l.made("layers.{d}.mlp.experts." ++ which ++ ".global_scale", .{i}, .{ .mul = 1.0 / nvfp4_span });
        return bk;
    }
};

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{"model.language_model"}, "embed_tokens.weight");
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    var l: Load = .{ .r = &r, .o = &m.owned };
    const hidden = c.hidden_size;
    const h: c_int = @intCast(hidden);
    m.layers = try gpa.alloc(Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(&l, c, i);

    if (c.vision) |v| m.tower = try vision.Tower.load(&r, &m.owned, v);

    const embed = try r.tableAt(try l.name("embed_tokens", .{}), hidden);
    try m.owned.linear(embed);
    const unembed = try r.tableAt("lm_head", hidden);
    try m.owned.linear(unembed);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = h, .act = .bfloat16 },
        .embed_norm = try l.need("embed_norm.weight", .{}),
        .layers = m.layers,
        .message_query = if (c.passes > 1) try l.made("message_input.query", .{}, .{}) else .{},
        .final_query = try l.made("final_input.query", .{}, .{ .sh = &.{ 1, h } }),
        .final_norm = try l.need("norm.weight", .{}),
        .head = .{ .lin = unembed, .vocab = @intCast(c.unpadded_vocab_size), .softcap = c.logit_softcap },
        .core_start = c.core_start,
        .core_end = c.core_end,
        .passes = c.passes,
        .eps = c.rms_norm_eps,
    };
    return m;
}

fn loadLayer(l: *Load, c: *const config.Config, i: usize) !Layer {
    const hidden = c.hidden_size;
    const h: c_int = @intCast(hidden);
    const global = c.global[i];
    const hs = c.heads(@intCast(i));
    const looped = c.passes > 1 and i >= c.core_start and i < c.core_end;
    const q_dim = hs.heads * hs.head_dim;

    var attn: attention.Weights = .{
        .q = try l.proj(hidden, "layers.{d}.self_attn.q_proj", .{i}),
        .k = try l.proj(hidden, "layers.{d}.self_attn.k_proj", .{i}),
        .v = try l.proj(hidden, "layers.{d}.self_attn.v_proj", .{i}),
        .rel = try l.proj(hidden, "layers.{d}.self_attn.r_proj", .{i}),
        .o = try l.proj(q_dim, "layers.{d}.self_attn.o_proj", .{i}),
        .gate = try l.proj(hidden, "layers.{d}.self_attn.gate_proj", .{i}),
        .sink = try l.proj(hidden, "layers.{d}.self_attn.sink_proj", .{i}),
        .q_clip = try l.need("layers.{d}.self_attn.q_l2_clip.scale", .{i}),
        .rel_clip = try l.need("layers.{d}.self_attn.rel_l2_clip.scale", .{i}),
        .rel_proj = try l.need("layers.{d}.self_attn.rel_logits_proj.proj", .{i}),
        .conv = undefined,
        .heads = @intCast(hs.heads),
        .kv_heads = @intCast(hs.kv_heads),
        .head_dim = @intCast(hs.head_dim),
        .d_rel = @intCast(c.d_rel),
        .window = if (global) 0 else c.sliding_window,
        .log_floor = c.log_n_floor,
        .log_alpha = c.log_alpha,
        .qk_clip = c.qk_clip,
        .rel_clip_at = c.rel_clip,
        .eps = c.rms_norm_eps,
    };
    attn.conv = if (global) .{ .dynamic = .{
        .k = try l.need("layers.{d}.self_attn.k_sconv.static_weight", .{i}),
        .v = try l.need("layers.{d}.self_attn.v_sconv.static_weight", .{i}),
        .k_proj = try l.proj(hidden, "layers.{d}.self_attn.k_sconv.weight_proj", .{i}),
        .v_proj = try l.proj(hidden, "layers.{d}.self_attn.v_sconv.weight_proj", .{i}),
    } } else .{ .static = .{
        .k = try l.need("layers.{d}.self_attn.k_sconv.conv1d.weight", .{i}),
        .v = try l.need("layers.{d}.self_attn.v_sconv.conv1d.weight", .{i}),
    } };

    var mlp: moe.Weights = .{
        .router = try l.made("layers.{d}.mlp.gate.weight", .{i}, .{ .t = true }),
        .bias = try l.made("layers.{d}.mlp.gate.e_score_correction_bias", .{i}, .{}),
        .gate_up = undefined,
        .down = undefined,
        .top_k = @intCast(c.top_k),
        .route_scale = c.route_scale,
        .gate_softcap = c.gate_softcap,
        .up_softcap = c.up_softcap,
    };
    mlp.gate_up = try l.bank(i, "gate_up_proj", hidden, &mlp.gate_up_scale);
    mlp.down = try l.bank(i, "down_proj", c.intermediate_size, &mlp.down_scale);

    var in_norm: [3]Array = undefined;
    if (global) {
        inline for (&in_norm, .{ "q", "k", "v" }) |*n, which| n.* = try l.need("layers.{d}." ++ which ++ "_input_layernorm.weight", .{i});
    } else in_norm = @splat(try l.need("layers.{d}.input_layernorm.weight", .{i}));

    return .{
        .global = global,
        .boundary = i % c.block_layers == 0,
        .attn = attn,
        .in_norm = in_norm,
        .attn_query = if (i > 0) try l.made("layers.{d}.attention_input.query", .{i}, .{ .sh = &.{ if (global) 3 else 1, h } }) else .{},
        .mlp_query = try l.made("layers.{d}.mlp_input.query", .{i}, .{ .sh = &.{ 1, h } }),
        .post_norm = try l.need("layers.{d}.post_attention_layernorm.weight", .{i}),
        .mlp = mlp,
        .attn_conv = try l.need("layers.{d}.post_attention_convolution.conv1d.weight", .{i}),
        .mlp_conv = try l.need("layers.{d}.post_mlp_convolution.conv1d.weight", .{i}),
        .gates = if (looped) .{ try l.gate(i, "q"), try l.gate(i, "k"), try l.gate(i, "v"), try l.gate(i, "mlp") } else null,
    };
}

/// The tiny pack the conformance suite runs: six layers of width 64 in blocks of two, layers 2..3 looped twice (eight
/// visits), layers 1 and 3 global, 4 experts of 32 (top 2), 60 of 64 vocab rows real; seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = sdk.fixture;
    const V = 64;
    const H = 64;
    const HD = 32;
    const HQ = 2;
    const E = 4;
    const MI = 32;
    const K = 4;
    const DR = 4;
    const R = 8;
    const RANK = 8;
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 97;
    const Put = struct {
        a: std.mem.Allocator,
        list: *std.ArrayList(fx.Tensor),
        seed: *u64,
        fn raw(p: @This(), name: []const u8, shape: []const u64, scale: f32, offset: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            p.seed.* += 1;
            const data = try fx.randomF32(p.a, p.seed.*, n, scale);
            for (data) |*x| x.* += offset;
            try p.list.append(p.a, .{ .name = name, .shape = try p.a.dupe(u64, shape), .data = data });
        }
        fn rand(p: @This(), comptime fmt: []const u8, args: anytype, shape: []const u64, scale: f32, offset: f32) !void {
            try p.raw(try std.fmt.allocPrint(p.a, "model.language_model." ++ fmt, args), shape, scale, offset);
        }
    };
    const p: Put = .{ .a = a, .list = &tensors, .seed = &seed };
    try p.rand("embed_tokens.weight", .{}, &.{ V, H }, 0.5, 0);
    try p.raw("lm_head.weight", &.{ V, H }, 0.5, 0);
    try p.rand("embed_norm.weight", .{}, &.{H}, 0.1, 1);
    try p.rand("norm.weight", .{}, &.{H}, 0.1, 1);
    try p.rand("final_input.query", .{}, &.{H}, 0.5, 0);
    try p.rand("message_input.query", .{}, &.{ 2, 4, H }, 0.5, 0);
    for (0..6) |i| {
        const global = i == 1 or i == 3;
        if (global) {
            inline for (.{ "q", "k", "v" }) |w| try p.rand("layers.{d}." ++ w ++ "_input_layernorm.weight", .{i}, &.{H}, 0.1, 1);
            inline for (.{ "k", "v" }) |w| {
                try p.rand("layers.{d}.self_attn." ++ w ++ "_sconv.static_weight", .{i}, &.{ K, HD }, 0.2, 0);
                try p.rand("layers.{d}.self_attn." ++ w ++ "_sconv.weight_proj.weight", .{i}, &.{ K, H }, 0.1, 0);
            }
        } else {
            try p.rand("layers.{d}.input_layernorm.weight", .{i}, &.{H}, 0.1, 1);
            inline for (.{ "k", "v" }) |w| try p.rand("layers.{d}.self_attn." ++ w ++ "_sconv.conv1d.weight", .{i}, &.{ HD, K, 1 }, 0.2, 0);
        }
        if (i > 0) try p.rand("layers.{d}.attention_input.query", .{i}, if (global) &.{ 3, H } else &.{H}, 0.5, 0);
        try p.rand("layers.{d}.mlp_input.query", .{i}, &.{H}, 0.5, 0);
        try p.rand("layers.{d}.post_attention_layernorm.weight", .{i}, &.{H}, 0.1, 1);
        try p.rand("layers.{d}.post_attention_convolution.conv1d.weight", .{i}, &.{ H, K, 1 }, 0.2, 0);
        try p.rand("layers.{d}.post_mlp_convolution.conv1d.weight", .{i}, &.{ H, K, 1 }, 0.2, 0);
        inline for (.{ .{ "q_proj", HQ * HD, H }, .{ "k_proj", HD, H }, .{ "v_proj", HD, H }, .{ "r_proj", HQ * DR, H }, .{ "o_proj", H, HQ * HD }, .{ "gate_proj", HQ * HD, H }, .{ "sink_proj", HQ, H } }) |w|
            try p.rand("layers.{d}.self_attn." ++ w[0] ++ ".weight", .{i}, &.{ w[1], w[2] }, 0.1, 0);
        try p.rand("layers.{d}.self_attn.q_l2_clip.scale", .{i}, &.{ HQ, HD }, 0.1, 1);
        try p.rand("layers.{d}.self_attn.rel_l2_clip.scale", .{i}, &.{ HQ, DR }, 0.1, 1);
        try p.rand("layers.{d}.self_attn.rel_logits_proj.proj", .{i}, &.{ DR, R }, 0.3, 0);
        try p.rand("layers.{d}.mlp.gate.weight", .{i}, &.{ E, H }, 0.2, 0);
        try p.rand("layers.{d}.mlp.gate.e_score_correction_bias", .{i}, &.{E}, 0.1, 0);
        try p.rand("layers.{d}.mlp.experts.gate_up_proj.weight", .{i}, &.{ E, 2 * MI, H }, 0.1, 0);
        try p.rand("layers.{d}.mlp.experts.down_proj.weight", .{i}, &.{ E, H, MI }, 0.1, 0);
        if (i == 2 or i == 3) inline for (.{ "q", "k", "v", "mlp" }) |w| {
            try p.rand("layers.{d}." ++ w ++ "_message_gate.down_proj.weight", .{i}, &.{ RANK, 2 * H }, 0.1, 0);
            try p.rand("layers.{d}." ++ w ++ "_message_gate.forget_gate_proj.weight", .{i}, &.{ H, RANK }, 0.3, 0);
            try p.rand("layers.{d}." ++ w ++ "_message_gate.message_gate_proj.weight", .{i}, &.{ H, RANK }, 0.3, 0);
        };
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"inkling2_loop","eos_token_id":2,"text_config":{"model_max_length":4096,"hidden_size":64,
        \\"num_hidden_layers":6,"vocab_size":64,"unpadded_vocab_size":60,"num_attention_heads":2,"num_key_value_heads":1,
        \\"head_dim":32,"swa_num_attention_heads":2,"swa_num_key_value_heads":1,"swa_head_dim":32,"d_rel":4,
        \\"rel_extent":8,"log_scaling_n_floor":2,"log_scaling_alpha":0.1,"rms_norm_eps":1e-6,
        \\"qk_l2_clip_threshold":5.0,"rel_l2_clip_threshold":2.0,"sconv_kernel_size":4,"final_logit_softcapping":30.0,
        \\"sliding_window_size":4,"n_layers_per_attn_res_block":2,"n_routed_experts":4,"num_experts_per_tok":2,
        \\"n_shared_experts":0,"intermediate_size":32,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":2.0,
        \\"local_layer_ids":[0,2,4,5],
        \\"loop_config":{"core_start":2,"core_end":4,"num_passes":2,"gate_rank":8,"is_varloop":false}}}
        },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
    _ = blocks;
    _ = attention;
    _ = vision;
}
