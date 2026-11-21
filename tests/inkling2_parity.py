# inkling2_parity.py <base url> <fixtures.json>: a running mlx-serve against tests/dump_inkling2_fixtures.py's oracle.
# Bar: the prompt ids are the reference's, and every top-10 logit of the last prefill row is within TOL of it (the
# reference's own Metal kernels and op chains differ by up to 0.5 there). The greedy text is printed, not gated: a
# near-tie flips it.
import json, sys, urllib.request, math
base, ref = sys.argv[1], json.load(open(sys.argv[2]))
def post(path, body):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
TOL = 0.75
ok = True
for name, e in ref.items():
    if "image_png_b64" in e:
        # A chat with the image first, at the reference's default effort. A chat returns no logprobs for its thinking
        # rows, so the bar is the prompt's length and a continuation that names what the picture holds.
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + e["image_png_b64"]}}, {"type": "text", "text": e["question"]}]
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 120, "temperature": 0, "reasoning_effort": "high", "stream": False})
        m = r["choices"][0]["message"]
        said = (m.get("reasoning_content") or "") + " | " + (m.get("content") or "")
        n = r["usage"]["prompt_tokens"]
        seen = [w for w in ("red", "green", "blue", "square", "circle") if w in said.lower()]
        good = n == len(e["ids"]) and len(seen) == 5
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {n} prompt tokens (reference {len(e['ids'])}), names {seen}")
        print("     ours:", repr(said[:160]))
        print("     ref :", repr(e["greedy_text"][:160]))
        continue
    ids = post("/tokenize", {"content": e["text"], "add_special": False})["tokens"]
    r = post("/v1/completions", {"prompt": e["text"], "max_tokens": 25, "temperature": 0, "logprobs": 10, "stream": False})
    c = r["choices"][0]
    top = c["logprobs"]["top_logprobs"][0]
    pairs = []
    for i, l in zip(e["top_ids"], e["top_logits"]):
        s = post("/detokenize", {"tokens": [i]})["content"]
        if s in top: pairs.append((l, top[s]))
    # Logprobs are logits less one constant: remove it by the mean.
    off = sum(l - o for l, o in pairs) / max(len(pairs), 1)
    worst = max((abs(o + off - l) for l, o in pairs), default=float("inf"))
    good = ids == e["ids"] and len(pairs) >= 8 and worst <= TOL
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {name}: {len(ids)} tokens, ids {'same' if ids == e['ids'] else 'DIFFER'}, {len(pairs)}/10 top tokens shared, worst logit gap {worst:.3f}")
    print("     ours:", repr(c["text"][:70]))
    print("     ref :", repr(e["greedy_text"][:70]))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
