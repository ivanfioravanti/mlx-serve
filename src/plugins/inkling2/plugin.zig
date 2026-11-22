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
const state = @import("../../sdk/bases/blocks/state.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");
const loader = @import("../llama/load.zig");
const attention = @import("attention.zig");
const blocks = @import("blocks.zig");
const compiled = @import("compiled.zig");
const moe = @import("moe.zig");
const vision = @import("vision.zig");
const host_abi = @import("../../abi/host.zig");
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
    .dist = &.{ .expert_share = &expertShare },
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
    /// `[4, hidden]` f32: a looped layer's q, k, v and mlp message queries.
    message_query: Array = .{},
    /// This rank's share of every expert (`--dist tensor`); null = a single node.
    share: ?moe.Half = null,
};

/// A tensor-parallel pair's state on rank 0: the host whose `expert_exchange` reaches the worker, and whether this
/// request still uses it (a failed exchange finishes the request on this node).
pub const Tp = struct {
    host: *const host_abi.Host,
    live: bool = true,
};

/// Prefill chunks from this many rows split their experts with the worker; decode, verify rounds and short spans
/// never cross the wire.
pub const TP_ROWS = 512;

pub const Head = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = true;
    pub const Weights = struct { lin: Linear, vocab: c_int, softcap: f32 };

    /// The padded vocab's real rows, softcapped in f32.
    pub fn forward(w: *const Weights, c: *block.Ctx, x: Array) !Array {
        const g = c.g;
        return g.sliceAxis(try compiled.softcap(g, try head_block.project(g, &w.lin, x), w.softcap), 2, 0, w.vocab);
    }
};

pub const Model = struct {
    pub const batches_decode = true;

    pub const Weights = struct {
        embed: b.embed.TableEmbed.Weights,
        embed_norm: Array,
        layers: []const Layer,
        final_query: Array,
        final_norm: Array,
        head: Head.Weights,
        core_start: u32,
        core_end: u32,
        passes: u32,
        eps: f32,
        /// Rank 0 of a tensor-parallel pair; null = a single node.
        tp: ?*Tp = null,
    };

    const Trunk = struct {
        w: *const Weights,
        c: *block.Ctx,
        g: *Graph,
        run: *const run_mod.Run,
        h: Array,
        /// `h` widened to f32 where a visit made it (empty = cast on use).
        h32: Array = .{},
        /// A residual a later pass restarts from: not dropped when the stream moves on.
        pinned: Array = .{},
        pinned32: Array = .{},
        bank: [64]Array = undefined,
        blocks: usize = 0,
        visit: u32 = 0,
        visits: u32,

        fn move(t: *Trunk, to: Array, to32: Array) !void {
            const old = t.h;
            const old32 = t.h32;
            t.h = try t.g.keep(to);
            t.h32 = try t.g.keep(to32);
            if (old.ctx != t.pinned.ctx) t.g.drop(old);
            if (old32.ctx != null and old32.ctx != t.pinned32.ctx) t.g.drop(old32);
        }

        /// The residual in f32, in `g`.
        fn wide(t: *const Trunk, g: *Graph) !Array {
            return if (t.h32.ctx != null) t.h32 else blocks.f32Of(g, t.h);
        }

        /// The MLP of a prefill chunk on a tensor-parallel pair: this rank's share of every expert, plus the worker's,
        /// sent before this share's graph so the return overlaps it; null = the single-node MLP (a short chunk, no
        /// pair, or the worker gone, which ends the pair for the request and is logged).
        fn pairMlp(t: *Trunk, L: *const Layer, li: usize, x: Array, x32: Array) !?Array {
            const tp = t.w.tp orelse return null;
            const share = L.share orelse return null;
            if (t.c.offset() == 0) tp.live = true;
            if (!tp.live or t.c.rows < TP_ROWS) return null;
            const h = tp.host;
            const exchange = host_abi.expertExchangeOf(h) orelse return null;
            const g = t.c.g;
            const sh = ops.shape(x);
            const flat = try g.reshape(x, &.{ sh[0] * sh[1], sh[2] });
            const r = try moe.routed(&L.mlp, g, flat, try g.reshape(x32, &.{ sh[0] * sh[1], sh[2] }));
            var theirs: host_abi.Array = .{};
            const v = exchange(h.ctx, @intCast(li), .{ .ctx = flat.ctx }, .{ .ctx = r.idx.ctx }, .{ .ctx = r.weights.ctx }, &theirs);
            if (!v.isOk()) {
                // No worker at all is the single-node server, not news.
                tp.live = false;
                if (v.code != .unsupported and h.log_enabled(h.ctx, .warn)) {
                    var buf: [160]u8 = undefined;
                    const line = std.fmt.bufPrint(&buf, "[inkling2] tensor-parallel worker lost at layer {d}: this request finishes on one node\n", .{li}) catch "";
                    h.log(h.ctx, .warn, host_abi.Str.of(line));
                }
                return null;
            }
            const mine = try moe.partial(&L.mlp, g, &share, flat, r);
            const sum = try g.add(mine, try g.adopt(.{ .ctx = theirs.ctx }));
            return try g.reshape(try g.astype(sum, ops.dtype(x)), sh);
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
            const normed_in = try blocks.rms(g, try t.wide(g), 1e-6);

            // The visit's conv windows: k | v | attention output | MLP output.
            const ns = attention.slots(c);
            const st = try state.get(c, .conv);
            var prev: [4]?Array = @splat(null);
            if (st) |a| {
                var at: c_int = 0;
                for (&prev, [_]c_int{ kvd, kvd, hidden, hidden }) |*p, n| {
                    p.* = try g.sliceAxis(a, 2, at, at + n);
                    at += n;
                }
            }

            var qkv: [3]Array = @splat(t.h);
            if (L.attn_query.ctx != null) {
                const bank = t.bank[0..t.blocks];
                if (L.global) {
                    const x = try blocks.depth(g, bank, normed_in, L.attn_query);
                    for (&qkv, 0..) |*s, j| s.* = try blocks.row(g, x, @intCast(j));
                } else qkv = @splat(try blocks.depthOne(g, bank, normed_in, L.attn_query));
            }
            if (messages) |m| for (&qkv, 0..) |*s, j| {
                s.* = try L.gates.?[j].onRow(g, s.*, m, @intCast(j));
            };
            var normed: [3]Array = undefined;
            for (&normed, qkv, 0..) |*n, s, j| {
                n.* = if (j > 0 and s.ctx == qkv[0].ctx and L.in_norm[j].ctx == L.in_norm[0].ctx) normed[0] else if (ops.dtype(s) == .float32 and ops.dtype(L.in_norm[j]) == .float32) (try blocks.rmsCast(g, s, L.in_norm[j], eps, dt, false)).out else try g.astype(try g.rmsNorm(s, L.in_norm[j], eps), dt);
            }
            const a = try attention.forward(&L.attn, c, normed[0], normed[1], normed[2], prev[0], prev[1]);
            const ac = try blocks.bySeq(g, try blocks.sconv(g, try blocks.bySlot(g, a.out, ns), L.attn_conv, prev[2]), ns);

            var res: Array = undefined;
            if (L.boundary) {
                res = try blocks.f32Of(g, ac.out);
                t.bank[t.blocks] = try t.g.keep(normed_in);
                t.blocks += 1;
            } else res = try compiled.sum(g, t.h, ac.out);

            var x = try blocks.depthOne(g, t.bank[0..t.blocks], try blocks.rms(g, res, 1e-6), L.mlp_query);
            if (messages) |m| x = try L.gates.?[3].onRow(g, x, m, 3);
            const mlp_in = try blocks.rmsCast(g, x, L.post_norm, eps, dt, true);
            const mc = if (try t.pairMlp(L, li, mlp_in.out, mlp_in.twin.?)) |y| try blocks.sconv(g, y, L.mlp_conv, prev[3]) else blk: {
                const parts = try moe.parts(&L.mlp, g, mlp_in.out, mlp_in.twin);
                // The experts' weighted sum rides the conv kernel where it serves; that kernel reads its rows as one
                // sequence, so a batched tick's slots take the sum and a conv each.
                const fused = if (ns == 1) try blocks.sconvSum(g, parts.out, parts.weights, L.mlp_conv, prev[3]) else null;
                break :blk fused orelse try blocks.bySeq(g, try blocks.sconv(g, try blocks.bySlot(g, try moe.sum(g, parts, ops.shape(mlp_in.out)), ns), L.mlp_conv, prev[3]), ns);
            };
            // One copy joins the four windows' last `K - 1` rows (views of the conv kernels' windows).
            try state.put(c, .conv, try g.concat(&.{ a.k.state, a.v.state, ac.state, mc.state }, 2));
            // A verify forward's rejected rows are rolled back from the conv inputs.
            if (c.capture_spec) try state.put(c, .conv_input, try g.concat(&.{ a.k.input, a.v.input, ac.input, mc.input }, 2));

            const next = try compiled.sumBoth(g, res, mc.out);
            try t.move(next[0], next[1]);
            try t.run.capture(t.g, t.visit, t.h);
            try t.run.cadence(t.visit, t.visits, t.h);
            t.visit += 1;
        }
    };

    /// One forward of `ids` (`[1, rows]` int32). Results are owned by `g`.
    pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: run_mod.Run) !run_mod.Out {
        if (run.mode == .batched_rows and run.rows != 1) return error.BatchedRowsUnsupported;
        var c = run.ctx(g, shared);
        const n_slots = attention.slots(&c);
        const embedded = try g.rmsNorm(try b.embed.TableEmbed.forward(&w.embed, &c, ids), w.embed_norm, w.eps);
        const n: u32 = @intCast(w.layers.len);
        const core = w.core_end - w.core_start;
        // A batched tick's slots ride the sequence axis: every row-wise block reads them as one chunk's rows.
        const input = try run.input(g, embedded, ids);
        const h0 = if (n_slots > 1) try g.reshape(input, &.{ 1, n_slots, ops.shape(input)[2] }) else input;
        var t: Trunk = .{ .w = w, .c = &c, .g = g, .run = &run, .h = h0, .visits = n + core * (w.passes - 1) };

        for (0..w.core_start) |i| try t.apply(i, null);
        t.pinned = t.h;
        t.pinned32 = t.h32;
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
            const last = try g.keep(try blocks.rms(g, try t.wide(g), 1e-6));
            if (t.h.ctx != t.pinned.ctx) g.drop(t.h);
            if (t.h32.ctx != null and t.h32.ctx != t.pinned32.ctx) g.drop(t.h32);
            t.h = t.pinned;
            t.h32 = t.pinned32;
            t.blocks = base;
            for (w.core_start..w.core_end) |i| {
                try t.apply(i, try blocks.depth(g, heard[0..said], last, w.layers[i].message_query));
            }
        }
        t.pinned = .{};
        t.pinned32 = .{};
        for (w.core_end..n) |i| try t.apply(i, null);

        const normed = try blocks.rms(g, try t.wide(g), 1e-6);
        const mixed = try blocks.depthOne(g, t.bank[0..t.blocks], normed, w.final_query);
        const normed_final = try g.astype(try g.rmsNorm(mixed, w.final_norm, w.eps), ops.dtype(t.h));
        const final = if (n_slots > 1) try g.reshape(normed_final, &.{ n_slots, 1, ops.shape(normed_final)[2] }) else normed_final;
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
    tp: Tp = undefined,

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

/// The worker's share of layer `layer`'s experts for rank 0's routes.
fn expertShare(mp: *anyopaque, layer: u32, x: abi.Array, idx: abi.Array, weights: abi.Array, out: *abi.Array) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (layer >= m.layers.len) return .{ .code = .failed, .why = abi.Str.of("expert share past the last layer") };
    const L = &m.layers[layer];
    const share = L.share orelse return .{ .code = .unsupported, .why = abi.Str.of("loaded without a tensor-parallel rank") };
    var g = Graph.init(gpa, m.base.s);
    defer g.deinit();
    const y = moe.partial(&L.mlp, &g, &share, .{ .ctx = x.ctx }, .{ .idx = .{ .ctx = idx.ctx }, .weights = .{ .ctx = weights.ctx } }) catch |e|
        return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    out.* = .{ .ctx = g.escape(y).ctx };
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

    /// Projections of one input, read unowned and joined row-wise (each `Linear` a view of its rows); kept apart where
    /// their quant geometry differs (an empty `Joined`).
    fn join(l: *Load, in_dim: u32, parts: []const *Linear, names: []const []const u8) !b.linear.Joined {
        for (parts, names) |p, n| p.* = try l.r.proj(n, in_dim);
        const j = if (parts[0].s.ctx != null) try loader.join(l.o, parts, l.r.s) else b.linear.Joined{};
        for (parts) |p| try l.o.linear(p.*);
        return j;
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

    /// Row `j` of an evaluated `[n, ...]` tensor, as its own evaluated copy.
    fn row(l: *Load, x: Array, j: c_int) !Array {
        var g = Graph.init(gpa, l.r.s);
        defer g.deinit();
        const y = try g.contiguous(try g.reshape(try g.sliceAxis(x, 0, j, j + 1), ops.shape(x)[1..]));
        try ops.eval(y);
        const kept = g.escape(y);
        try l.o.own(kept);
        return kept;
    }

    fn gate(l: *Load, i: usize, comptime which: []const u8) !blocks.MessageGate {
        return .{
            .down = try l.made("layers.{d}." ++ which ++ "_message_gate.down_proj.weight", .{i}, .{ .dt = null }),
            .forget = try l.made("layers.{d}." ++ which ++ "_message_gate.forget_gate_proj.weight", .{i}, .{}),
            .message = try l.made("layers.{d}." ++ which ++ "_message_gate.message_gate_proj.weight", .{i}, .{}),
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

    if (c.passes > 1) {
        const all = try l.made("message_input.query", .{}, .{});
        for (m.layers[c.core_start..c.core_end], 0..) |*L, j| L.message_query = try l.row(all, @intCast(j));
    }
    if (c.vision) |v| m.tower = try vision.Tower.load(&r, &m.owned, v);

    const embed = try r.tableAt(try l.name("embed_tokens", .{}), hidden);
    try m.owned.linear(embed);
    const unembed = try r.tableAt("lm_head", hidden);
    try m.owned.linear(unembed);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = h, .act = .bfloat16 },
        .embed_norm = try l.need("embed_norm.weight", .{}),
        .layers = m.layers,
        .final_query = try l.made("final_input.query", .{}, .{ .sh = &.{ 1, h } }),
        .final_norm = try l.made("norm.weight", .{}, .{}),
        .head = .{ .lin = unembed, .vocab = @intCast(c.unpadded_vocab_size), .softcap = c.logit_softcap },
        .core_start = c.core_start,
        .core_end = c.core_end,
        .passes = c.passes,
        .eps = c.rms_norm_eps,
    };
    if (ctx.tp_size > 0) {
        // Each rank cuts its own share; rank 0 keeps the full banks for decode and a request the worker left.
        var g = Graph.init(gpa, r.s);
        defer g.deinit();
        for (m.layers) |*L| {
            const s = try moe.half(&L.mlp, &g, @intCast(ctx.tp_rank), @intCast(ctx.tp_size));
            for ([_]Array{ s.gate_up.w, s.gate_up.s, s.gate_up.b, s.down.w, s.down.s, s.down.b }) |a| if (a.ctx != null) try m.owned.own(a);
            L.share = s;
        }
        if (ctx.tp_rank == 0) if (ctx.host) |hp| {
            m.tp = .{ .host = hp };
            m.base.weights.tp = &m.tp;
        };
    }
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
        .q = .{},
        .k = .{},
        .v = .{},
        .rel = .{},
        .o = try l.proj(q_dim, "layers.{d}.self_attn.o_proj", .{i}),
        .gate = .{},
        .sink = .{},
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
    var nb: [4][256]u8 = undefined;
    const r = l.r;
    attn.join_q = try l.join(hidden, &.{ &attn.q, &attn.rel, &attn.gate, &attn.sink }, &.{
        try r.name(&nb[0], "layers.{d}.self_attn.q_proj", .{i}),
        try r.name(&nb[1], "layers.{d}.self_attn.r_proj", .{i}),
        try r.name(&nb[2], "layers.{d}.self_attn.gate_proj", .{i}),
        try r.name(&nb[3], "layers.{d}.self_attn.sink_proj", .{i}),
    });
    if (global) {
        var kp: Linear = .{};
        var vp: Linear = .{};
        attn.join_k = try l.join(hidden, &.{ &attn.k, &kp }, &.{ try r.name(&nb[0], "layers.{d}.self_attn.k_proj", .{i}), try r.name(&nb[1], "layers.{d}.self_attn.k_sconv.weight_proj", .{i}) });
        attn.join_v = try l.join(hidden, &.{ &attn.v, &vp }, &.{ try r.name(&nb[0], "layers.{d}.self_attn.v_proj", .{i}), try r.name(&nb[1], "layers.{d}.self_attn.v_sconv.weight_proj", .{i}) });
        attn.conv = .{ .dynamic = .{
            .k = try l.need("layers.{d}.self_attn.k_sconv.static_weight", .{i}),
            .v = try l.need("layers.{d}.self_attn.v_sconv.static_weight", .{i}),
            .k_proj = kp,
            .v_proj = vp,
        } };
    } else {
        attn.join_kv = try l.join(hidden, &.{ &attn.k, &attn.v }, &.{ try r.name(&nb[0], "layers.{d}.self_attn.k_proj", .{i}), try r.name(&nb[1], "layers.{d}.self_attn.v_proj", .{i}) });
        attn.conv = .{ .static = .{
            .k = try l.need("layers.{d}.self_attn.k_sconv.conv1d.weight", .{i}),
            .v = try l.need("layers.{d}.self_attn.v_sconv.conv1d.weight", .{i}),
        } };
    }

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

    // f32 where the input is (every layer but 0, whose input is the bf16 embedding), so no forward casts them.
    const norm_dt: ?mlx.mlx_dtype = if (i > 0) .float32 else null;
    var in_norm: [3]Array = undefined;
    if (global) {
        inline for (&in_norm, .{ "q", "k", "v" }) |*n, which| n.* = try l.made("layers.{d}." ++ which ++ "_input_layernorm.weight", .{i}, .{ .dt = norm_dt });
    } else in_norm = @splat(try l.made("layers.{d}.input_layernorm.weight", .{i}, .{ .dt = norm_dt }));

    return .{
        .global = global,
        .boundary = i % c.block_layers == 0,
        .attn = attn,
        .in_norm = in_norm,
        .attn_query = if (i > 0) try l.made("layers.{d}.attention_input.query", .{i}, .{ .sh = &.{ if (global) 3 else 1, h } }) else .{},
        .mlp_query = try l.made("layers.{d}.mlp_input.query", .{i}, .{ .sh = &.{ 1, h } }),
        .post_norm = try l.made("layers.{d}.post_attention_layernorm.weight", .{i}, .{}),
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
    _ = @import("l2_clip.zig");
    _ = @import("sconv.zig");
    _ = @import("rms_cast.zig");
    _ = @import("experts.zig");
    _ = moe;
}

const conformance = @import("../../sdk/testing/conformance.zig");
const fake_host = @import("../../sdk/testing/fake_host.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");

/// The worker on the same process: its share of a layer, or a link that fails at `fail_at`.
const Pair = struct {
    var worker: *anyopaque = undefined;
    var fail_at: ?u32 = null;
    var present: bool = true;
    var calls: u32 = 0;

    fn exchange(_: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, w: abi.Array, out: *abi.Array) callconv(.c) abi.Verdict {
        if (!present) return .{ .code = .unsupported, .why = abi.Str.of("no tensor-parallel worker") };
        calls += 1;
        if (fail_at == layer) return .{ .code = .failed, .why = abi.Str.of("link down") };
        return table.dist.?.expert_share.?(worker, layer, x, idx, w, out);
    }

    fn prefill(a: std.mem.Allocator, pack: *conformance.Loaded, tokens: []const u32) ![]f32 {
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        const h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        const v = table.forward(pack.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
        if (!v.isOk()) return error.ForwardRefused;
        if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
        return conformance.consumeF32(a, out.logits);
    }
};

test "inkling2 pair: rank 0's split plus the worker's share is one node's prefill; no worker is its bytes; a lost one falls back and logs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var fh: fake_host.FakeHost = undefined;
    fh.init(.{ .allocator = a, .level = .warn });
    fh.table.expert_exchange = &Pair.exchange;
    var whole = try conformance.Loaded.open(io, a, plugin.name, &table, try conformanceFixture(a));
    defer whole.close();
    var worker = try conformance.Loaded.openWith(io, a, plugin.name, &table, try conformanceFixture(a), .{ .tp_rank = 1, .tp_size = 2 });
    defer worker.close();
    var rank0 = try conformance.Loaded.openWith(io, a, plugin.name, &table, try conformanceFixture(a), .{ .host = fh.host(), .tp_rank = 0, .tp_size = 2 });
    defer rank0.close();
    Pair.worker = worker.model;

    const tokens = try a.alloc(u32, TP_ROWS);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 7 + 1) % 60);
    const want = try Pair.prefill(a, &whole, tokens);
    // Bar: f32 partial sums in another order, through eight visits.
    const near = struct {
        fn check(x: []const f32, y: []const f32) !void {
            for (x, y) |p, q| try std.testing.expectApproxEqAbs(p, q, 2e-3);
        }
    };

    Pair.calls = 0;
    try near.check(want, try Pair.prefill(a, &rank0, tokens));
    try std.testing.expectEqual(@as(u32, 8), Pair.calls);
    // A chunk under the threshold never crosses the wire.
    Pair.calls = 0;
    _ = try Pair.prefill(a, &rank0, tokens[0 .. TP_ROWS - 1]);
    try std.testing.expectEqual(@as(u32, 0), Pair.calls);

    Pair.present = false;
    try std.testing.expectEqualSlices(f32, want, try Pair.prefill(a, &rank0, tokens));
    try std.testing.expectEqual(@as(usize, 0), fh.lines);
    Pair.present = true;

    Pair.fail_at = 3;
    try near.check(want, try Pair.prefill(a, &rank0, tokens));
    try std.testing.expect(std.mem.indexOf(u8, fh.last(), "worker lost at layer 3") != null);
    Pair.fail_at = null;
}
