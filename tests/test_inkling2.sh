#!/bin/bash
# test_inkling2.sh — Inkling 2 Mini (inkling2_loop) against mlx-lm's reference: serves the pack and holds the last
# prefill row's top-10 logits to the oracle (tests/inkling2_parity.py), then checks that greedy text with PLD, and with
# the pack's DSpark drafter when mtp.safetensors sits beside it (thinkingmachines/Inkling-2-Mini ships the file), still
# continues a self-repeating list while drafts are accepted (a partial accept rolls every visit's conv windows back).
#
#   INKLING2_TEST_MODEL=<pack dir> INKLING2_FIXTURES=<json from tests/dump_inkling2_fixtures.py> ./tests/test_inkling2.sh
#
# INKLING2_TEST_PORT (11358). SKIPs without both env vars.

set -euo pipefail

MODEL="${INKLING2_TEST_MODEL:-}"
FIXTURES="${INKLING2_FIXTURES:-}"
if [ -z "$MODEL" ] || [ -z "$FIXTURES" ]; then
    echo "SKIP: INKLING2_TEST_MODEL and INKLING2_FIXTURES not both set"
    exit 0
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${INKLING2_TEST_PORT:-11358}"
BASE="http://127.0.0.1:$PORT"
BIN="$HERE/../zig-out/bin/mlx-serve"
PID=""
trap '[ -n "$PID" ] && kill "$PID" 2>/dev/null; true' EXIT

serve() {
    LOG=$(mktemp /tmp/inkling2_serve.XXXXXX)
    "$BIN" --model "$MODEL" --port "$PORT" --log-level debug "$@" >"$LOG" 2>&1 &
    PID=$!
    for _ in $(seq 1 180); do
        curl -s -m 2 "$BASE/health" | grep -q '"ok"' && return 0
        kill -0 "$PID" 2>/dev/null || { echo "FAIL: server exited"; tail -5 "$LOG"; exit 1; }
        sleep 2
    done
    echo "FAIL: server not ready"
    exit 1
}

stop() {
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    PID=""
}

# Verify rows and a decode row round differently, so a tie between two tokens can fall the other way and drafted text
# is not held to serial decode's: the bar is that the continuation of a list it copies is still the list.
still_list() {
    python3 - "$BASE" <<'PY'
import json, re, sys, urllib.request
rivers = ["Po", "Tiber", "Adige", "Arno", "Piave", "Reno", "Sarca", "Mincio"]
prompt = "".join(f"{i}. The river {n} flows through the old town, past the stone bridge and the mill, and on to the sea.\n" for i, n in enumerate(rivers, 1)) + "9. The river"
req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({"prompt": prompt, "max_tokens": 120, "temperature": 0, "stream": False}).encode(), {"Content-Type": "application/json"})
text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]
lines = [l for l in ("9. The river" + text).split("\n") if l][:3]
if len(lines) < 3 or not all(re.match(r"\d+\. The river \w+ flows through the old town, past the stone bridge and the mill", l) for l in lines):
    sys.exit("FAIL: the drafted continuation left the list: " + repr(text[:160]))
PY
}

serve --no-drafter
python3 "$HERE/inkling2_parity.py" "$BASE" "$FIXTURES"
still_list
grep -q "mode=pld attempts=[0-9]* accepts=[1-9]" "$LOG" || { echo "FAIL: PLD never had a draft accepted (log $LOG)"; exit 1; }
stop
echo "PASS: PLD drafts accepted, the list continues"

# The pack's own drafter (mtp.safetensors): most drafts accepted, the list continues.
if [ -e "$MODEL/mtp.safetensors" ]; then
    serve --no-pld
    still_list
    RATE=$(sed -n 's/.*mode=dflash attempts=[0-9]* accepts=[0-9]* avg_per_round=\([0-9.]*\).*/\1/p' "$LOG" | tail -1)
    stop
    if [ -z "$RATE" ] || ! awk -v r="$RATE" 'BEGIN { exit !(r >= 2.0) }'; then
        echo "FAIL: the drafter had ${RATE:-no} drafts accepted per round, expected at least 2 (log $LOG)"
        exit 1
    fi
    echo "PASS: the drafter had $RATE drafts accepted per round"
else
    echo "SKIP: no mtp.safetensors beside the pack (the drafter)"
fi
