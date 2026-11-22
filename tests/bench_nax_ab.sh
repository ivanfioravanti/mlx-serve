#!/bin/bash
# bench_nax_ab.sh — same-session A/B for the MLX NAX (tensor-ops) build swap.
#
# Replicates tests/bench.sh's exact metrics (salted prompts, first-run warmup
# drop, prefill = prompt_toks/wall, decode = _decode_stream.py rate) against a
# manually booted mlx-serve with the calibrated ddalcu Qwen3.6-27B MTP
# checkpoint, so a libmlx rebuild is the ONLY variable between runs.
#
# Usage:  bash tests/bench_nax_ab.sh <label> [out_csv]
# Example: bash tests/bench_nax_ab.sh before /tmp/nax_ab.csv
set -euo pipefail

LABEL="${1:?label (before|after|...)}"
OUT="${2:-/tmp/nax_ab_$(date +%Y%m%d).csv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT/zig-out/bin/mlx-serve"
MODEL="$HOME/.mlx-serve/models/ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve"
PORT=11250
CTX=4096
PREFILL_RUNS=4   # first is warmup → 3 timed
DECODE_RUNS=3    # first is warmup → 2 timed

# Same prompts as tests/bench.sh (lines 388-389).
PREFILL_PROMPT="Explain the following topics in extreme detail: $(python3 -c "print(', '.join([f'topic {i} about science and technology and its impact on human civilization throughout history' for i in range(1,50)]))")"
DECODE_PROMPT="Write a detailed essay about quantum computing"
salted() { echo "[run-$1-$RANDOM] $2"; }

body() { # prompt max_tokens
    jq -nc --arg p "$1" --argjson mt "$2" \
        '{model:"x", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, enable_thinking:false, enable_pld:false, enable_drafter:false, enable_mtp:false}'
}

[[ -x "$BINARY" ]] || { echo "missing $BINARY" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "missing model $MODEL" >&2; exit 1; }

"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size "$CTX" \
    --log-level info >/tmp/bench_nax_ab_server.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
    kill -0 "$PID" 2>/dev/null || { echo "server died; see /tmp/bench_nax_ab_server.log" >&2; exit 1; }
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server never healthy" >&2; exit 1; }

# ── Prefill (max_tokens=1, wall time → tps; drop first run) ──
elapsed_csv="" pt=0
for i in $(seq 1 "$PREFILL_RUNS"); do
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(body "$(salted "$i" "$PREFILL_PROMPT")" 1 | curl -sf -m 300 -X POST \
        "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d @-)
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    pt=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['usage']['prompt_tokens'])" "$resp")
    [[ "$i" -gt 1 ]] && elapsed_csv+="$((t1-t0)),"
done
prefill_tps=$(python3 -c "
e=[float(x) for x in '${elapsed_csv%,}'.split(',') if x]
avg=sum(e)/len(e)
print(f'{$pt/(avg/1000.0):.1f}' if avg>0 else '0')")

# ── Decode (stream first→last token rate; drop first run) ──
rates=()
for i in $(seq 1 "$DECODE_RUNS"); do
    out=$(body "$(salted "$i" "$DECODE_PROMPT")" 128 | \
        python3 "$SCRIPT_DIR/_decode_stream.py" "http://127.0.0.1:$PORT/v1/chat/completions" 240)
    IFS='|' read -r rate ntok rn <<<"$out"
    [[ "$i" -gt 1 && "$rate" != "ERR" && "$rate" != "0" ]] && rates+=("$rate")
done
decode_tps=$(python3 -c "
r=[float(x) for x in '${rates[*]:-0}'.split()]
print(f'{sum(r)/len(r):.1f}' if r else '0')")

mkdir -p "$(dirname "$OUT")"
grep -q '^label|' "$OUT" 2>/dev/null || \
    echo "label|prefill_tps|decode_tps|prompt_toks|ctx|date" >> "$OUT"
echo "$LABEL|$prefill_tps|$decode_tps|$pt|$CTX|$(date +%F\ %T)" | tee -a "$OUT"
