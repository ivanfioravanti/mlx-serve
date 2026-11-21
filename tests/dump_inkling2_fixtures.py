# dump_inkling2_fixtures.py <pack dir> <out.json>: the oracle tests/test_inkling2.sh compares a server against.
# Needs mlx-lm with the inkling2 model (PR #2086 until it merges). Per prompt: the ids, the last row's top-10 logits
# after a whole-prompt prefill, and a greedy continuation fed one token at a time. The prompts cover a short one, the
# chat template, one past the sliding window (1024) and one past the log-scaling floor (8192, two prefill chunks).
import json, sys
import mlx.core as mx
from mlx_lm import load

path, out = sys.argv[1], sys.argv[2]
model, tok = load(path)
cases = {
    "short": "The capital of France is",
    "chat": tok.apply_chat_template([{"role": "user", "content": "Write a haiku about rivers."}], tokenize=False, add_generation_prompt=True),
    "long": "The quick brown fox jumps over the lazy dog. " * 150,
    "xlong": "".join(f"{i}. Entry {i * 7919 % 1000} of the ledger was checked against line {i * 31 % 977} and filed.\n" for i in range(520)),
}
res = {}
for name, text in cases.items():
    ids = tok.encode(text, add_special_tokens=False)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    top = mx.argsort(-logits)[:10].tolist()
    entry = {"text": text, "ids": ids, "top_ids": top, "top_logits": [logits[i].item() for i in top]}
    gen = []
    nxt = int(mx.argmax(logits).item())
    for _ in range(24):
        gen.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
        nxt = int(mx.argmax(logits).item())
    entry["greedy_ids"] = gen
    entry["greedy_text"] = tok.decode(gen)
    res[name] = entry
    print(name, len(ids), repr(entry["greedy_text"]), top[:3], entry["top_logits"][:3], flush=True)
json.dump(res, open(out, "w"))
