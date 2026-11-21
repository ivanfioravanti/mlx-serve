# inkling2_parity.py <base url> <fixtures.json>: a running mlx-serve against tests/dump_inkling2_fixtures.py's oracle.
# Bar: the prompt ids are the reference's; of the last prefill row's top-10, at least 5 are shared (to fit the
# logprobs' offset), every shared logit is within the prompt's bar, and our top-1 is the reference's unless the
# reference's own top-2 margin is under the bar. A prompt's bar is max(0.75, its self_gap): the reference's Metal
# kernels and its own op chains disagree by that much on the same prompt (up to 1.5 on a 9.9k-token one), and a
# rounding-equivalent change here moves a long prompt's worst gap as far. The greedy text is printed, not gated.
import json, sys, urllib.request, math
base, ref = sys.argv[1], json.load(open(sys.argv[2]))
def post(path, body):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
FLOOR = 0.75
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
    bar = max(FLOOR, e.get("self_gap", 0.0))
    pairs = []
    ref_top1 = post("/detokenize", {"tokens": [e["top_ids"][0]]})["content"]
    for i, l in zip(e["top_ids"], e["top_logits"]):
        s = post("/detokenize", {"tokens": [i]})["content"]
        if s in top: pairs.append((l, top[s]))
    # Logprobs are logits less one constant: remove it by the mean.
    off = sum(l - o for l, o in pairs) / max(len(pairs), 1)
    gaps = [abs(o + off - l) for l, o in pairs]
    worst = max(gaps, default=float("inf"))
    mean = sum(gaps) / max(len(gaps), 1)
    ours_top1 = max(top, key=top.get)
    margin = e["top_logits"][0] - e["top_logits"][1]
    if ours_top1 == ref_top1:
        top1, top1_ok = "top-1 same", True
    elif margin < bar:
        top1, top1_ok = f"top-1 differs inside the reference's own top-2 margin {margin:.3f}", True
    else:
        top1, top1_ok = f"top-1 DIFFERS ({ours_top1!r} vs {ref_top1!r}, reference margin {margin:.3f})", False
    good = ids == e["ids"] and len(pairs) >= 5 and worst <= bar and top1_ok
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {name}: {len(ids)} tokens, ids {'same' if ids == e['ids'] else 'DIFFER'}, {len(pairs)}/10 top tokens shared, worst gap {worst:.3f} mean {mean:.3f} (bar {bar:.3f}), {top1}")
    print("     ours:", repr(c["text"][:70]))
    print("     ref :", repr(e["greedy_text"][:70]))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
