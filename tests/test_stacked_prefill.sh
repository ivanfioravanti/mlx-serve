#!/bin/bash
# test_stacked_prefill.sh — admitted cold prompts prefilled stacked (`prefill_stacked.zig`, a model declaring
# Caps.stacks_prompts): [1] four cold prompts sent at once each decode what they decode alone after (raw completions,
# greedy; a first divergence at the alone run's near-tie is acquitted, as the batched rows round apart); [2] the pass
# engaged for them (its log line); [3] a lone request never stacks, and one after the stack is no restore.
#   [4] four requests behind one cached prefix (a primed system prompt, each its own tail) stack their restored tails in
#   one pass and each decodes what it decodes alone.
#   With MLX_SERVE_DIAG=stacked-prefill-fail the pass fails after its first slot: [2] becomes its failure line and [1]
#   must still hold (every slot reset and prefilled alone).
#
#   MLX_STACK_MODEL=<dir> ./tests/test_stacked_prefill.sh   # default Inkling-2-Mini-4bit; PORT (19078)

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
source "$(dirname "$0")/_lib_models.sh"
MODEL="${MLX_STACK_MODEL:-$(find_model mlx-community/Inkling-2-Mini-4bit)}"
PORT="${PORT:-19078}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"
[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "fail: build mlx-serve first"; exit 1; }

srv_preclean "$PORT"
LOG="$(mktemp)"
SERVER_PID=""
trap 'kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; true' EXIT
"$BIN" --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
srv_wait_health "$BASE" 180 0.5 "$SERVER_PID" || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
SLOG="$HOME/.mlx-serve/logs/mlx-serve-$PORT.log"

# The stacked rows round apart from a lone prompt's (the batched-equivalence bar; inkling2's is its verify rows' class).
TIE_NATS=0.15
[ "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]+'/config.json')).get('model_type',''))" "$MODEL")" = "inkling2_loop" ] && TIE_NATS=0.75
python3 - "$BASE" "$SLOG" "$TIE_NATS" <<'PY'
import json, sys, threading, urllib.request
base, slog, tie = sys.argv[1], sys.argv[2], float(sys.argv[3])
words = "river lantern harbor ledger meadow canyon glacier thunder cabin orchard signal copper marble ribbon".split()
def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
def call(prompt, lp=True):
    # A logprobs request never stacks (it decodes serial): only the alone runs carry them, for the acquittal. The others
    # decode plain too (a stacked slot's drafter is off, and a failed pass's slots must compare like for like).
    body = {"prompt": prompt, "max_tokens": 24, "temperature": 0, "stream": False, "enable_drafter": False}
    if lp: body["logprobs"] = 2
    c = post("/v1/completions", body)["choices"][0]
    return c["logprobs"] if lp else {"tokens": [str(t) for t in post("/tokenize", {"content": c["text"]})["tokens"]]}
def prompt(i, tag):
    return f"[{tag} {i}] " + " ".join(f"{words[(i * 5 + k) % len(words)]}{k % 7}" for k in range(60 + 9 * i)) + ". Next:"
import os
fail = "stacked-prefill-fail" in os.environ.get("MLX_SERVE_DIAG", "")
needle = "[prefill] stacked pass failed" if fail else "[prefill] stacked "
def lines():
    try: return sum(needle in l for l in open(slog))
    except OSError: return 0
ec = 0
before = lines()
call(prompt(9, "lone"))
if lines() != before:
    print("❌ FAIL: a lone request stacked"); ec = 1
else:
    print("✅ a lone request prefills alone")
# Stacked first, while the prompts are cold; then each alone (a prefix restore, whose class the bound acquits too).
got = [None] * 4
th = [threading.Thread(target=lambda i=i: got.__setitem__(i, call(prompt(i, "stack"), False))) for i in range(4)]
before = lines()
[t.start() for t in th]; [t.join() for t in th]
# A repeat of a stacked prompt restores where the pass left its checkpoint: the snapshot backoff, 31 rows short of its
# end, so its tail re-runs as the cold prompt's final span did (no model declares the end-1 cut).
stacked_cut = tie >= 0.75 and not fail
def restores():
    try: return [l for l in open(slog) if "[hot-cache] restore pos=" in l]
    except OSError: return []
seen = len(restores())
rep = post("/v1/completions", {"prompt": prompt(0, "stack"), "max_tokens": 1, "temperature": 0})
n_prompt = rep.get("usage", {}).get("prompt_tokens", 0)
cached = (rep.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
if stacked_cut:
    if cached != n_prompt - 31:
        print(f"❌ FAIL: a repeat of a stacked prompt restored {cached} of {n_prompt} (want {n_prompt - 31}: the backoff cut)"); ec = 1
    else:
        print("✅ a repeat of a stacked prompt restores at the pass's backoff cut")
alone = [call(prompt(i, "stack")) for i in range(4)]
for a in alone:
    a["ids"] = [str(t) for t in post("/tokenize", {"content": "".join(a["tokens"])})["tokens"]]
if lines() == before:
    print(f"❌ FAIL: four prompts at once never logged {needle!r}"); ec = 1
else:
    print("✅ the stacked pass " + ("failed and reset" if fail else "engaged"))
after = post("/v1/completions", {"prompt": prompt(11, "after"), "max_tokens": 4, "temperature": 0})
if (after.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) != 0:
    print("❌ FAIL: a fresh prompt after the stack was taken as restored"); ec = 1
else:
    print("✅ a fresh prompt after the stack prefills cold")
def compare(alone, got, what):
    global ec
    for i in range(len(alone)):
        a, b = alone[i], got[i]
        for k in range(min(len(a["ids"]), len(b["tokens"]))):
            if a["ids"][k] != b["tokens"][k]:
                top = sorted((a["top_logprobs"][k] or {}).values(), reverse=True)
                gap = top[0] - top[1] if len(top) > 1 else 99.0
                ok = gap <= tie
                print(f"{'✅' if ok else '❌ FAIL:'} {what} {i}: {'near-tie acquitted' if ok else 'diverged'} at token {k} (alone top-2 gap {gap:.3f} nats, bound {tie})")
                ec |= not ok
                break
        else:
            print(f"✅ {what} {i}: stacked decodes what it decodes alone")
compare(alone, got, "prompt")
# [4] One cached prefix, four tails: the prefix primed alone, then the four at once (their restored tails stack), then
# each alone (a restore of its own tail).
prefix = "[shared] " + " ".join(f"{words[(k * 3) % len(words)]}{k % 11}" for k in range(700)) + "."
def tailed(i):
    return prefix + f" [user {i}] " + " ".join(f"{words[(i * 7 + k) % len(words)]}{k % 5}" for k in range(40 + 6 * i)) + ". Next:"
call(prefix, False)
warm_needle = "[prefill] stacked pass failed" if fail else "restored tails in one forward"
def warm_lines():
    try: return sum(warm_needle in l for l in open(slog))
    except OSError: return 0
wgot = [None] * 4
th = [threading.Thread(target=lambda i=i: wgot.__setitem__(i, call(tailed(i), False))) for i in range(4)]
before = warm_lines()
[t.start() for t in th]; [t.join() for t in th]
walone = [call(tailed(i)) for i in range(4)]
for a in walone:
    a["ids"] = [str(t) for t in post("/tokenize", {"content": "".join(a["tokens"])})["tokens"]]
if warm_lines() == before:
    print(f"❌ FAIL: four tails behind one cached prefix never logged {warm_needle!r}"); ec = 1
else:
    print("✅ the restored tails " + ("failed their pass and reset" if fail else "stacked"))
compare(walone, wgot, "tail")
sys.exit(ec)
PY
