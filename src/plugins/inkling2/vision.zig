//! Inkling 2's image tower (mlx-lm `inkling2/vision.py`, `processor.py`): an image at its own size, padded to whole
//! merge blocks, as rows of patches in block order; a ViT with a class token, a bilinearly interpolated position
//! grid, 2D RoPE (row and column, offset by one for the class token) and per-layer residual gains; the class token
//! is dropped and each block's patches merge into one row of the text width.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const loader = @import("../llama/load.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const config = @import("config.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

const rope_theta: f32 = 10000;

/// A dense projection stored `[in, out]`; `b` null = no bias.
const Dense = struct {
    w: Array,
    b: Array = .{},

    fn forward(d: Dense, g: *Graph, x: Array) !Array {
        const y = try g.matmul(x, d.w);
        return if (d.b.ctx != null) g.add(y, d.b) else y;
    }
};

const Layer = struct {
    norm1: [2]Array,
    norm2: [2]Array,
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    fc1: Dense,
    fc2: Dense,
    gain1: Array,
    gain2: Array,
};

fn gelu(g: *Graph, x: Array) !Array {
    const x32 = try g.astype(x, .float32);
    const cdf = try g.mulScalar(try g.addScalar(try g.erf(try g.mulScalar(x32, std.math.sqrt1_2)), 1), 0.5);
    return g.astype(try g.mul(x32, cdf), ops.dtype(x));
}

/// A patch index's row and column in the patch grid: blocks in raster order, then each block's patches.
fn patchAt(i: usize, blocks_w: usize, merge: usize) [2]usize {
    const blk = i / (merge * merge);
    const in = i % (merge * merge);
    return .{ blk / blocks_w * merge + in / merge, blk % blocks_w * merge + in % merge };
}

/// The two taps and weights of position `index` of `size` on a grid of `side` (half-pixel centres, edges clamped).
const Taps = struct { at: [2]i32, w: [2]f32 };

fn taps(index: usize, size: usize, side: usize) Taps {
    const src = (@as(f32, @floatFromInt(index)) + 0.5) * @as(f32, @floatFromInt(side)) / @as(f32, @floatFromInt(size)) - 0.5;
    const lo = @floor(src);
    var out: Taps = undefined;
    for (0..2) |t| {
        const tap = lo + @as(f32, @floatFromInt(t));
        out.w[t] = @max(1.0 - @abs(src - tap), 0.0);
        out.at[t] = @intFromFloat(std.math.clamp(tap, 0, @as(f32, @floatFromInt(side - 1))));
    }
    return out;
}

pub const Tower = struct {
    cfg: config.Vision,
    patch: Dense,
    /// `[grid * grid, hidden]`.
    positions: Array,
    /// The class token with its position, `[1, hidden]`.
    class: Array,
    pre_norm: [2]Array,
    layers: []Layer,
    merge1: Dense,
    merge2: Dense,
    s: ops.Stream,

    pub fn deinit(t: *Tower) void {
        gpa.free(t.layers);
    }

    /// The tower of the pack, or null where it ships none. Arrays are `owned`'s.
    pub fn load(r: *loader.Reader, owned: *loader.Owned, cfg: config.Vision) !?Tower {
        const root = "model.vision_tower.";
        if (!r.has(root ++ "pre_layernorm.weight")) return null;
        var l: Load = .{ .r = r, .o = owned };
        const class = try l.get(root ++ "embeddings.class_embedding", .{}, .plain);
        const class_pos = try l.get(root ++ "embeddings.cls_position_embedding", .{}, .plain);
        var g = Graph.init(gpa, r.s);
        defer g.deinit();
        const cls = try g.reshape(try g.add(class, class_pos), &.{ 1, @intCast(cfg.hidden) });
        try ops.eval(cls);
        const cls_kept = g.escape(cls);
        try owned.own(cls_kept);
        var t: Tower = .{
            .cfg = cfg,
            .patch = .{ .w = try l.get(root ++ "embeddings.patch_embedding.weight", .{}, .transposed) },
            .positions = try l.get(root ++ "embeddings.position_embedding.weight", .{}, .plain),
            .class = cls_kept,
            .pre_norm = .{ try l.get(root ++ "pre_layernorm.weight", .{}, .plain), try l.get(root ++ "pre_layernorm.bias", .{}, .plain) },
            .layers = try gpa.alloc(Layer, cfg.layers),
            .merge1 = .{ .w = try l.get(root ++ "merger.linear_1.weight", .{}, .transposed) },
            .merge2 = .{ .w = try l.get(root ++ "merger.linear_2.weight", .{}, .transposed) },
            .s = r.s,
        };
        errdefer t.deinit();
        for (t.layers, 0..) |*L, i| {
            const at = root ++ "layers.{d}.";
            L.* = .{
                .norm1 = .{ try l.get(at ++ "layernorm_before.weight", .{i}, .plain), try l.get(at ++ "layernorm_before.bias", .{i}, .plain) },
                .norm2 = .{ try l.get(at ++ "layernorm_after.weight", .{i}, .plain), try l.get(at ++ "layernorm_after.bias", .{i}, .plain) },
                .q = try l.dense(at ++ "self_attn.q_proj", i),
                .k = try l.dense(at ++ "self_attn.k_proj", i),
                .v = try l.dense(at ++ "self_attn.v_proj", i),
                .o = try l.dense(at ++ "self_attn.out_proj", i),
                .fc1 = try l.dense(at ++ "mlp.fc1", i),
                .fc2 = try l.dense(at ++ "mlp.fc2", i),
                .gain1 = try l.get(at ++ "lambda_1", .{i}, .plain),
                .gain2 = try l.get(at ++ "lambda_2", .{i}, .plain),
            };
        }
        return t;
    }

    const Load = struct {
        r: *loader.Reader,
        o: *loader.Owned,
        nb: [200]u8 = undefined,

        fn get(l: *Load, comptime fmt: []const u8, args: anytype, how: enum { plain, transposed }) !Array {
            const a = try l.r.need(try std.fmt.bufPrint(&l.nb, fmt, args));
            if (how == .plain) {
                try l.o.own(a);
                return a;
            }
            defer _ = mlx.mlx_array_free(a);
            var g = Graph.init(gpa, l.r.s);
            defer g.deinit();
            const t = try g.contiguous(try g.transpose(a, &.{ 1, 0 }));
            try ops.eval(t);
            const kept = g.escape(t);
            try l.o.own(kept);
            return kept;
        }

        fn dense(l: *Load, comptime base: []const u8, i: usize) !Dense {
            return .{ .w = try l.get(base ++ ".weight", .{i}, .transposed), .b = try l.get(base ++ ".bias", .{i}, .plain) };
        }
    };

    /// `pixels` `[gh * gw, patch * patch * 3]` f32 in block order: the image's rows `[gh / merge * gw / merge, text
    /// hidden]`, a new reference.
    pub fn forward(t: *const Tower, pixels: Array, gh: u32, gw: u32) !Array {
        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const c = t.cfg;
        const n: usize = @as(usize, gh) * gw;
        const hidden: c_int = @intCast(c.hidden);
        const heads: c_int = @intCast(c.heads);
        const hd: usize = c.hidden / c.heads;
        const side: usize = c.posemb_grid;
        const dt = ops.dtype(t.patch.w);

        // Per patch: its four position taps and weights; per token (class first, at 0, the patches offset by one):
        // the RoPE angles, columns over the first half of the head, rows over the second, each frequency twice.
        const at = try gpa.alloc(i32, n * 4);
        defer gpa.free(at);
        const wts = try gpa.alloc(f32, n * 4);
        defer gpa.free(wts);
        const cos = try gpa.alloc(f32, (n + 1) * hd);
        defer gpa.free(cos);
        const sin = try gpa.alloc(f32, (n + 1) * hd);
        defer gpa.free(sin);
        @memset(cos[0..hd], 1);
        @memset(sin[0..hd], 0);
        const half = hd / 2;
        for (0..n) |i| {
            const rc = patchAt(i, gw / c.merge, c.merge);
            const tr = taps(rc[0], gh, side);
            const tc = taps(rc[1], gw, side);
            for (0..2) |a| for (0..2) |b| {
                at[i * 4 + a * 2 + b] = tr.at[a] * @as(i32, @intCast(side)) + tc.at[b];
                wts[i * 4 + a * 2 + b] = tr.w[a] * tc.w[b];
            };
            for (0..half / 2) |j| {
                const inv = std.math.pow(f32, rope_theta, -@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(half)));
                const angles = [2]f32{ @as(f32, @floatFromInt(rc[1] + 1)) * inv, @as(f32, @floatFromInt(rc[0] + 1)) * inv };
                for (angles, 0..) |angle, part| for (0..2) |rep| {
                    const k = (i + 1) * hd + part * half + j * 2 + rep;
                    cos[k] = @cos(angle);
                    sin[k] = @sin(angle);
                };
            }
        }
        const tok: c_int = @intCast(n + 1);
        const cos_a = try g.fromSlice(f32, cos, &.{ tok, 1, @intCast(hd) });
        const sin_a = try g.fromSlice(f32, sin, &.{ tok, 1, @intCast(hd) });

        const taken = try g.reshape(try g.take(t.positions, try g.fromSlice(i32, at, &.{@intCast(n * 4)}), 0), &.{ @intCast(n), 4, hidden });
        const pos = try g.sum(try g.mul(try g.astype(taken, .float32), try g.fromSlice(f32, wts, &.{ @intCast(n), 4, 1 })), 1, false);
        const embedded = try g.add(try t.patch.forward(&g, try g.astype(pixels, dt)), try g.astype(pos, dt));
        var h = try g.layerNorm(try g.concat(&.{ try g.astype(t.class, dt), embedded }, 0), t.pre_norm[0], t.pre_norm[1], c.eps);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (t.layers) |*L| {
            var lg = g.scope();
            defer lg.deinit();
            const x = try lg.layerNorm(h, L.norm1[0], L.norm1[1], c.eps);
            const q = try heads4(&lg, try rope(&lg, try lg.reshape(try L.q.forward(&lg, x), &.{ tok, heads, -1 }), cos_a, sin_a));
            const k = try heads4(&lg, try rope(&lg, try lg.reshape(try L.k.forward(&lg, x), &.{ tok, heads, -1 }), cos_a, sin_a));
            const v = try heads4(&lg, try lg.reshape(try L.v.forward(&lg, x), &.{ tok, heads, -1 }));
            const a = try lg.reshape(try lg.transpose(try lg.sdpa(q, k, v, scale, .none), &.{ 0, 2, 1, 3 }), &.{ tok, hidden });
            const mid = try lg.add(h, try lg.mul(L.gain1, try L.o.forward(&lg, a)));
            const y = try L.fc2.forward(&lg, try gelu(&lg, try L.fc1.forward(&lg, try lg.layerNorm(mid, L.norm2[0], L.norm2[1], c.eps))));
            const next = try g.keep(try lg.add(mid, try lg.mul(L.gain2, y)));
            g.drop(h);
            h = next;
        }
        const blocks: c_int = @intCast(n / (c.merge * c.merge));
        const merged = try g.reshape(try g.sliceAxis(h, 0, 1, tok), &.{ blocks, -1 });
        const out = try t.merge2.forward(&g, try gelu(&g, try t.merge1.forward(&g, merged)));
        return g.escape(out);
    }
};

/// `[tokens, heads, d]` as attention reads it, `[1, heads, tokens, d]`.
fn heads4(g: *Graph, x: Array) !Array {
    return g.expandDims(try g.transpose(x, &.{ 1, 0, 2 }), 0);
}

/// Rotates each pair of `x` `[tokens, heads, d]` by its angle (`cos`/`sin` `[tokens, 1, d]`), in f32.
fn rope(g: *Graph, x: Array, cos: Array, sin: Array) !Array {
    const sh = ops.shape(x);
    const st = try g.astype(x, .float32);
    const pairs = try g.reshape(st, &.{ sh[0], sh[1], @divExact(sh[2], 2), 2 });
    const turned = try g.reshape(try g.concat(&.{ try g.neg(try g.sliceAxis(pairs, 3, 1, 2)), try g.sliceAxis(pairs, 3, 0, 1) }, 3), sh);
    return g.astype(try g.add(try g.mul(st, cos), try g.mul(turned, sin)), ops.dtype(x));
}

/// The patch rows of an RGB8 image as the processor lays them out: the image at the top left of whole merge blocks,
/// the padding -1 on the 0..255 scale, every value `(x / 255 - 0.5) / 0.5`. An image over `budget` pixels is resized
/// to fit first (the reference never resizes).
pub const Patches = struct { values: []f32, gh: u32, gw: u32 };

pub fn patches(a: std.mem.Allocator, rgb: []const u8, height: u32, width: u32, v: config.Vision, budget: u64) !Patches {
    if (height == 0 or width == 0 or rgb.len != @as(usize, height) * width * 3) return error.InvalidImageBuffer;
    var h = height;
    var w = width;
    var unit: ?qwen_vision.UnitImage = null;
    defer if (unit) |u| a.free(u.pixels);
    if (budget > 0 and @as(u64, h) * w > budget) {
        const f = @sqrt(@as(f64, @floatFromInt(budget)) / @as(f64, @floatFromInt(@as(u64, h) * w)));
        h = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(h)) * f)));
        w = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(w)) * f)));
        unit = try qwen_vision.resizeToUnitChw(a, rgb, height, width, h, w);
    }
    const planes: ?[]const f32 = if (unit) |u| @as([*]const f32, @ptrCast(@alignCast(u.pixels.ptr)))[0 .. @as(usize, h) * w * 3] else null;
    const block = v.block();
    const rows = (h + block - 1) / block;
    const cols = (w + block - 1) / block;
    const p: usize = v.patch;
    const m: usize = v.merge;
    const feat = p * p * 3;
    const values = try a.alloc(f32, @as(usize, rows) * cols * m * m * feat);
    const scale: f32 = 1.0 / 255.0;
    for (0..@as(usize, rows) * cols * m * m) |i| {
        const rc = patchAt(i, cols, m);
        for (0..p) |py| for (0..p) |px| for (0..3) |ch| {
            const y = rc[0] * p + py;
            const x = rc[1] * p + px;
            const raw: f32 = if (y >= h or x >= w)
                -scale
            else if (planes) |pl|
                pl[ch * @as(usize, h) * w + y * w + x]
            else
                @as(f32, @floatFromInt(rgb[(y * w + x) * 3 + ch])) * scale;
            values[i * feat + (py * p + px) * 3 + ch] = (raw - 0.5) / 0.5;
        };
    }
    return .{ .values = values, .gh = rows * v.merge, .gw = cols * v.merge };
}

pub fn prepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    const v = config.of(p).vision orelse return Verdict.refuse(.unsupported, "inkling2: this pack has no vision tower");
    if (in.kind != .image or in.frames > 1) return Verdict.refuse(.unsupported, "inkling2: images only");
    const px = patches(gpa, in.rgb.slice(), in.height, in.width, v, budget) catch return Verdict.refuse(.failed, "inkling2: the image has no patch grid");
    defer gpa.free(px.values);
    const feat: c_int = @intCast(v.patch * v.patch * 3);
    out.* = .{
        .kind = .image,
        .rows = px.gh / v.merge * (px.gw / v.merge),
        .grid = .{ 1, px.gh, px.gw },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(px.values.ptr, &[_]c_int{ @intCast(px.gh * px.gw), feat }, 2, .float32).ctx },
    };
    return .ok;
}

pub fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *@import("plugin.zig").Loaded = @ptrCast(@alignCast(mp));
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "inkling2: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "inkling2: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

// ── tests ──

const testing = std.testing;

test "inkling2 vision: a 50 x 90 image pads to 2 x 3 blocks, patches in block order, padding at -1/255" {
    const v: config.Vision = .{};
    const rgb = try testing.allocator.alloc(u8, 50 * 90 * 3);
    defer testing.allocator.free(rgb);
    for (0..50) |y| for (0..90) |x| {
        rgb[(y * 90 + x) * 3 + 0] = @intCast(y);
        rgb[(y * 90 + x) * 3 + 1] = @intCast(x);
        rgb[(y * 90 + x) * 3 + 2] = 255;
    };
    const px = try patches(testing.allocator, rgb, 50, 90, v, 0);
    defer testing.allocator.free(px.values);
    try testing.expectEqual(@as(u32, 4), px.gh);
    try testing.expectEqual(@as(u32, 6), px.gw);
    const norm = struct {
        fn f(x: f32) f32 {
            return (x / 255.0 - 0.5) / 0.5;
        }
    }.f;
    // Patch 5 is block 1's second patch: rows 0..19, columns 60..79; its pixel (3, 7) is image pixel (3, 67).
    const at = 5 * 1200 + (3 * 20 + 7) * 3;
    try testing.expectApproxEqAbs(norm(3), px.values[at], 1e-6);
    try testing.expectApproxEqAbs(norm(67), px.values[at + 1], 1e-6);
    try testing.expectApproxEqAbs(norm(255), px.values[at + 2], 1e-6);
    // Patch 14 is block 3 (second block row), patch (1, 0): rows 60..79, past the image.
    try testing.expectApproxEqAbs(norm(-1), px.values[14 * 1200], 1e-6);
    try testing.expectEqual([2]usize{ 3, 5 }, patchAt(23, 3, 2));
}

test "inkling2 vision: position taps sit on the grid at its own size and clamp at the edges" {
    const same = taps(5, 32, 32);
    try testing.expectEqual(@as(i32, 5), same.at[0]);
    try testing.expectApproxEqAbs(@as(f32, 1), same.w[0], 1e-6);
    const edge = taps(0, 64, 32);
    try testing.expectEqual(@as(i32, 0), edge.at[0]);
    try testing.expectEqual(@as(i32, 0), edge.at[1]);
    try testing.expectApproxEqAbs(@as(f32, 1), edge.w[0] + edge.w[1], 1e-6);
}
