#!/bin/bash
# test_stacked_prefill.sh — admitted cold prompts prefilled stacked (`prefill_stacked.zig`, a model declaring
# Caps.stacks_prompts): [1] four prompts sent at once each decode what they decode alone (raw completions, greedy;
# a first divergence at the alone run's near-tie is acquitted, as the batched rows round apart); [2] the stacked pass
# engaged for them (its log line); [3] a lone request never stacks.
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
def call(prompt):
    body = {"prompt": prompt, "max_tokens": 24, "temperature": 0, "stream": False, "logprobs": 2}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["logprobs"]
def prompt(i, tag):
    return f"[{tag} {i}] " + " ".join(f"{words[(i * 5 + k) % len(words)]}{k % 7}" for k in range(60 + 9 * i)) + ". Next:"
def lines():
    try: return sum("[prefill] stacked" in l for l in open(slog))
    except OSError: return 0
ec = 0
before = lines()
call(prompt(9, "lone"))
if lines() != before:
    print("❌ FAIL: a lone request stacked"); ec = 1
else:
    print("✅ a lone request prefills alone")
alone = [call(prompt(i, "alone")) for i in range(4)]
got = [None] * 4
th = [threading.Thread(target=lambda i=i: got.__setitem__(i, call(prompt(i, "stack")))) for i in range(4)]
before = lines()
[t.start() for t in th]; [t.join() for t in th]
if lines() == before:
    print("❌ FAIL: four prompts at once never stacked (no '[prefill] stacked' line)"); ec = 1
else:
    print("✅ the stacked pass engaged")
for i in range(4):
    a, b = alone[i], got[i]
    for k in range(min(len(a["tokens"]), len(b["tokens"]))):
        if a["tokens"][k] != b["tokens"][k]:
            top = sorted((a["top_logprobs"][k] or {}).values(), reverse=True)
            gap = top[0] - top[1] if len(top) > 1 else 99.0
            ok = gap <= tie
            print(f"{'✅' if ok else '❌ FAIL:'} prompt {i}: {'near-tie acquitted' if ok else 'diverged'} at token {k} (alone top-2 gap {gap:.3f} nats, bound {tie})")
            ec |= not ok
            break
    else:
        print(f"✅ prompt {i}: stacked decodes what it decodes alone")
sys.exit(ec)
PY
