# dump_inkling2_drafter_fixtures.py <pack dir> <oMLX checkout> <out.json>: the oracle for Inkling 2's DSpark drafter.
# The trunk is mlx-lm's (PR #2086 until it merges); the drafter is oMLX's module (omlx/patches/inkling2/dspark.py,
# loaded from the checkout) over the pack's mtp.safetensors, driven here one row at a time with vLLM's context
# windows.
# Writes one draft round for the env-gated Zig test (the taps, the block's input rows, its normed hidden rows and
# the proposed ids) and, per prompt, the drafts the trunk's own greedy continuation accepts per round: the bar a
# served acceptance is held to.
import importlib.util, json, sys, types
import mlx.core as mx
from mlx_lm import load
import mlx_lm.models.inkling2.layers as L

pack, omlx, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(omlx + "/omlx/patches/inkling2/dspark.py").read().replace("from . import INKLING2_MODEL_TYPES", 'INKLING2_MODEL_TYPES = frozenset({"inkling2", "inkling2_loop"})')
D = types.ModuleType("omlx_dspark")
sys.modules["omlx_dspark"] = D
exec(compile(src, "dspark.py", "exec"), D.__dict__)

model, tok = load(pack)
lm = model.model.language_model
dcfg = D.Inkling2DSparkConfig.from_target_config(json.load(open(pack + "/config.json")))
draft = D.Inkling2DSparkModel(dcfg)
draft.load_weights(list(mx.load(pack + "/mtp.safetensors").items()), strict=True)
draft.bind(types.SimpleNamespace(model=types.SimpleNamespace(embed_norm=lm.embed_norm, embed=lm.embed_tokens), lm_head=model.lm_head))

# A tap is the residual entering a layer visit (visits in call order); the last one is the final-norm output.
taps_want = dcfg.target_layer_ids[:-1]
calls = {"n": 0, "got": {}}
orig = L.DecoderLayer.__call__
def hooked(self, hidden_states, block_states, **kw):
    if calls["n"] in taps_want:
        calls["got"][calls["n"]] = hidden_states
    calls["n"] += 1
    return orig(self, hidden_states, block_states, **kw)
L.SlidingDecoderLayer.__call__ = hooked
L.FullDecoderLayer.__call__ = hooked

def taps_for(ids):
    calls["n"], calls["got"] = 0, {}
    final = lm(mx.array([ids]))
    ctx = mx.concatenate([calls["got"][i].astype(mx.bfloat16) for i in taps_want] + [final.astype(mx.bfloat16)], axis=-1)
    mx.eval(ctx)
    return ctx

def greedy(ids, n):
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)[0, -1]
    out = []
    for _ in range(n):
        out.append(int(mx.argmax(logits).item()))
        logits = model(mx.array([[out[-1]]]), cache=cache)[0, -1]
    return out

def draft_round(ctx, anchor, pos):
    """`ctx` [1, n, taps * hidden]: the taps of positions < pos; the anchor token sits at `pos`."""
    ctx = ctx[:, :pos]
    block = dcfg.block_size
    h = draft._embed_input_tokens(mx.array([[anchor]]))
    h = mx.concatenate([h, mx.broadcast_to(draft.mask_embed.astype(h.dtype), (1, block - 1, h.shape[-1]))], axis=1)
    noise = h
    for index, layer in enumerate(draft.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)
        # As vLLM's drafter: a local layer keeps the newest `sliding_window` context rows, the others all of them.
        lo = max(0, pos - dcfg.sliding_window) if index in dcfg.local_layer_ids else 0
        n = pos - lo
        ck, cv = a.project_context_kv(ctx[:, lo:])
        ck = draft.rope(a.k_norm_ctx(ck.reshape(1, n, a.n_kv_heads, a.head_dim)).transpose(0, 2, 1, 3), offset=lo)
        cv = cv.reshape(1, n, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        q = draft.rope(a.q_norm(a.q_proj(x).reshape(1, block, a.n_heads, -1)).transpose(0, 2, 1, 3), offset=pos)
        k = draft.rope(a.k_norm(a.k_proj(x).reshape(1, block, a.n_kv_heads, -1)).transpose(0, 2, 1, 3), offset=pos)
        v = a.v_proj(x).reshape(1, block, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
        att = mx.fast.scaled_dot_product_attention(q, mx.concatenate([ck, k], axis=2), mx.concatenate([cv, v], axis=2), scale=a.scale)
        h = h + a.o_proj(att.transpose(0, 2, 1, 3).reshape(1, block, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    normed = draft.norm(h)
    logits = draft._logits(normed[:, 1:])
    draft.markov_head.prepare_block(normed[:, 1:], ctx, [pos], ["row"])
    prev, ids = mx.array([anchor]), []
    for step in range(block - 1):
        ids.append(int(mx.argmax(draft.markov_head.apply_step_logits(logits[:, step], prev)[0]).item()))
        prev = mx.array([ids[-1]])
    return ids, noise, normed

flat = lambda a: a.astype(mx.float32).reshape(-1).tolist()
prompts = {
    "code": "Write a Python function that parses an ISO-8601 date string without using datetime, with a few doctest examples.",
    "prose": "Explain in three paragraphs why the sky is blue and sunsets are red.",
    "list": "List the planets of the solar system with one fact each.",
}
res = {"hidden_size": dcfg.hidden_size, "n_targets": len(dcfg.target_layer_ids), "block_size": dcfg.block_size, "prompts": {}}
for name, prompt in prompts.items():
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    every = ids + greedy(ids, 160)
    ctx = taps_for(every)
    if "ctx_stream" not in res:
        # One round at the first generated position, for the Zig test.
        pos = len(ids)
        proposed, noise, normed = draft_round(ctx, every[pos], pos)
        res.update(n_ctx=pos, anchor=every[pos], ctx_stream=flat(ctx[:, :pos]), noise1=flat(noise), round1_hidden=flat(normed), proposed=proposed)
    p, rounds, accepted = len(ids), 0, 0
    while p + dcfg.block_size <= len(every):
        proposed, _, _ = draft_round(ctx, every[p], p)
        k = 0
        while k < len(proposed) and proposed[k] == every[p + 1 + k]:
            k += 1
        rounds, accepted, p = rounds + 1, accepted + k, p + k + 1
    res["prompts"][name] = {"prompt": prompt, "accepted_per_round": accepted / rounds}
    print(name, f"{accepted / rounds:.2f} drafts accepted per round over {rounds} rounds", flush=True)
json.dump(res, open(out_path, "w"))
