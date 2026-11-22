#!/bin/bash
# nax_subbyte_probe.sh — compile-and-run wrapper for nax_subbyte_probe.mm.
#
# Probes MetalPerformancePrimitives matmul2d (NAX path) for sub-byte weight
# formats. Reads the actual __HAVE_* macro names and sub-byte type spellings
# out of the SDK headers and forwards them to the probe as -D defines, scans
# the block-scaled ABI surface, then builds and runs the probe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVELOPER_DIR="/Applications/Xcode-beta.app/Contents/Developer"
export DEVELOPER_DIR
SDK="$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.sdk"
HEADERS="$SDK/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers"
PUB="$HEADERS/MPPTensorOpsMatMul2d.h"
IMPL="$HEADERS/__impl/MPPTensorOpsMatMul2dImpl.h"
BUILD_DIR="$SCRIPT_DIR/.build"
mkdir -p "$BUILD_DIR"

if [ ! -f "$PUB" ]; then
  echo "FATAL: $PUB not found (SDK layout changed?)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# SDK header scan: extract feature-macro names and sub-byte type spellings.
# A name that is absent from the headers is reported "header-absent" and its
# define is simply not passed to the probe.
# ---------------------------------------------------------------------------

echo "== SDK header scan =="
echo "headers: $HEADERS"

find_macro() { # $1 = exact macro token; prints it if present in the headers
  grep -rhoE "$1" "$HEADERS" 2>/dev/null | sort -u | sed -n '1p' || true
}
find_type() { # $1 = word-boundaried type token
  grep -rhoE "[[:<:]]$1[[:>:]]" "$HEADERS" 2>/dev/null | sort -u | sed -n '1p' || true
}

M_TENSOR=$(find_macro '__HAVE_TENSOR__')
M_TENSOR_MP=$(find_macro '__HAVE_TENSOR_MULTIPLANE__')
M_BFLOAT=$(find_macro '__HAVE_BFLOAT__')
M_INT4=$(find_macro '__HAVE_INT4B_FORMAT_TYPE__')
M_INT2=$(find_macro '__HAVE_INT2B_FORMAT_TYPE__')
M_FP4=$(find_macro '__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__')
M_FP8E4M3=$(find_macro '__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__')
M_FP8E5M2=$(find_macro '__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__')

T_UINT4=$(find_type 'uint4b_format')
T_INT4=$(find_type 'int4b_format')
T_UINT2=$(find_type 'uint2b_format')
T_INT2=$(find_type 'int2b_format')
T_FP4=$(find_type 'metal_fp4_e2m1_format')
T_FP8E4M3=$(find_type 'metal_fp8_e4m3_format')
T_FP8E5M2=$(find_type 'metal_fp8_e5m2_format')
T_FP8UE8M0=$(find_type 'metal_fp8_ue8m0_format')

report() { # $1 label, $2 value
  if [ -n "$2" ]; then printf '  %-34s %s\n' "$1" "$2"; else printf '  %-34s %s\n' "$1" 'ABSENT'; fi
}
echo "-- feature macros --"
report '__HAVE_TENSOR__' "$M_TENSOR"
report '__HAVE_TENSOR_MULTIPLANE__' "$M_TENSOR_MP"
report '__HAVE_BFLOAT__' "$M_BFLOAT"
report '__HAVE_INT4B_FORMAT_TYPE__' "$M_INT4"
report '__HAVE_INT2B_FORMAT_TYPE__' "$M_INT2"
report '__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__' "$M_FP4"
report '__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__' "$M_FP8E4M3"
report '__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__' "$M_FP8E5M2"
echo "-- sub-byte type spellings (namespace metal::) --"
report 'uint4b_format' "$T_UINT4"
report 'int4b_format' "$T_INT4"
report 'uint2b_format' "$T_UINT2"
report 'int2b_format' "$T_INT2"
report 'metal_fp4_e2m1_format' "$T_FP4"
report 'metal_fp8_e4m3_format' "$T_FP8E4M3"
report 'metal_fp8_e5m2_format' "$T_FP8E5M2"
# Report-only: MX block-scale factor type. Present in the impl datatype enum
# but NOT in the public matmul2d combo table — no ladder entry.
report 'metal_fp8_ue8m0_format (report-only, no ladder entry)' "$T_FP8UE8M0"

DEFS=()

# Per-format defines. fp16/int8 controls use standard Metal spellings; their
# gate macro is the tensor-API macro found above.
emit_format() { # $1 fmt-key, $2 macro-value, $3 type-value
  if [ -n "$2" ] && [ -n "$3" ]; then
    DEFS+=("-DPROBE_MACRO_$1=$2" "-DPROBE_TYPE_$1=$3")
  else
    echo "  header-absent: $1 (macro='${2:-none}' type='${3:-none}') — compiled without its defines"
  fi
}
echo "-- format define wiring --"
emit_format FP16  "$M_TENSOR" half
emit_format INT8  "$M_TENSOR" int8_t
emit_format UINT4 "$M_INT4"   "$T_UINT4"
emit_format INT4  "$M_INT4"   "$T_INT4"
emit_format FP4   "$M_FP4"    "$T_FP4"
emit_format INT2  "$M_INT2"   "$T_INT2"
emit_format FP8E4M3 "$M_FP8E4M3" "$T_FP8E4M3"
emit_format FP8E5M2 "$M_FP8E5M2" "$T_FP8E5M2"

# All discovered macros, indexed, for the probe's summary dump.
i=0
for mv in "$M_TENSOR" "$M_TENSOR_MP" "$M_BFLOAT" "$M_INT4" "$M_INT2" "$M_FP4" "$M_FP8E4M3" "$M_FP8E5M2"; do
  if [ -n "$mv" ]; then
    DEFS+=("-DPROBE_MACRO_$i=$mv")
    i=$((i + 1))
  fi
done
DEFS+=("-DPROBE_MACRO_COUNT=$i")
echo

# ---------------------------------------------------------------------------
# Block-scaled ABI scan
# ---------------------------------------------------------------------------

echo "== Block-scaled ABI scan =="

# Public matmul2d_descriptor body — any scale fields?
DESC_BODY=$(sed -n '/struct matmul2d_descriptor/,/^};/p' "$PUB")
if printf '%s\n' "$DESC_BODY" | grep -qiE 'scale'; then
  PUB_DESC_SCALE=yes
else
  PUB_DESC_SCALE=no
fi
# Any scale mention anywhere in the public matmul2d header?
PUB_SCALE_LINES=$(grep -niE 'scale' "$PUB" || true)

echo "block-scaled descriptors in public ABI: $PUB_DESC_SCALE"
if [ "$PUB_DESC_SCALE" = yes ]; then
  printf '%s\n' "$DESC_BODY" | grep -iE 'scale' | sed 's/^/    /'
fi
if [ -n "$PUB_SCALE_LINES" ]; then
  echo "  note: public header 'scale' mentions (comments/identifiers):"
  printf '%s\n' "$PUB_SCALE_LINES" | sed 's/^/    /'
fi

# Impl run symbols carrying scale params.
ALL_RUN=$(grep -oE '__tensorops_impl_matmul2d_op_run_[A-Za-z0-9_]*' "$IMPL" | sort -u)
SCALE_RUN=$(grep -E '__tensorops_impl_matmul2d_op_run_[A-Za-z0-9_]*' "$IMPL" \
            | grep -E 'Scale(DataType|BlockSize)' \
            | grep -oE '__tensorops_impl_matmul2d_op_run_[A-Za-z0-9_]*' | sort -u)
N_ALL=$(printf '%s\n' "$ALL_RUN" | grep -c . || true)
N_SCALE=$(printf '%s\n' "$SCALE_RUN" | grep -c . || true)

if [ "$N_SCALE" -gt 0 ]; then
  echo "scale params in impl run symbols: yes ($N_SCALE of $N_ALL unique run symbols carry leftScale*/rightScale* params)"
  echo "  distinct operand-type combos among scale-carrying symbols:"
  printf '%s\n' "$SCALE_RUN" \
    | sed -E 's/__tensorops_impl_matmul2d_op_run_(cooperative_)?//; s/_(dv|tg)_/ /g; s/_v2$//' \
    | sort -u | sed -n '1,40p' | sed 's/^/    /'
  echo "  first 20 scale-carrying symbol names (deduped):"
  printf '%s\n' "$SCALE_RUN" | sed -n '1,20p' | sed 's/^/    /'
  echo "  ... ($N_SCALE total)"
else
  echo "scale params in impl run symbols: no (0 of $N_ALL unique run symbols)"
fi
echo

# ---------------------------------------------------------------------------
# Compile and run
# ---------------------------------------------------------------------------

echo "== Build =="
echo "defines: ${DEFS[*]}"
xcrun --sdk macosx clang++ \
  -mmacosx-version-min=26.2 -fobjc-arc -O1 \
  -framework Metal -framework Foundation \
  "${DEFS[@]}" \
  "$SCRIPT_DIR/nax_subbyte_probe.mm" -o "$BUILD_DIR/nax_subbyte_probe"
echo "built: $BUILD_DIR/nax_subbyte_probe"
echo

echo "== Run =="
set +e
"$BUILD_DIR/nax_subbyte_probe" "$@"
rc=$?
set -e
exit "$rc"
