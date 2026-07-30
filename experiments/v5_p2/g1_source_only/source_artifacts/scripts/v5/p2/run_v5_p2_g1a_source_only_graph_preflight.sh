#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/inspect_v5_p2_g1a_source_only_graph_preflight.py"
OUT="$REPO/reports/v5/p2_g1a_source_only_graph_baseline_preflight"
LOG="$REPO/logs/v5/v5_p2_g1a_source_only_graph_baseline_preflight.log"

echo "===== V5 P2-G1A SOURCE-ONLY GRAPH PREFLIGHT ====="
echo "repo=$REPO"
echo "g0=$G0"
echo "data=$DATA"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo "checkpoint_loaded=false"
echo "b4_validation_cache_accessed=false"
echo "b6_test_cache_accessed=false"
echo "training_performed=false"
echo "optimization_steps=0"
echo "architecture_selected=false"
echo "quantization_performed=false"
echo "rtl_generated=false"
echo "legal_decoder_implemented=false"
echo

for path in "$G0" "$DATA"; do
    [ -d "$path" ] || {
        echo "STOP: missing directory: $path"
        exit 2
    }
done

for path in "$LOADER" "$MODEL" "$SCRIPT"; do
    [ -f "$path" ] || {
        echo "STOP: missing file: $path"
        exit 2
    }
done

[ ! -e "$OUT" ] || {
    echo "STOP: output already exists: $OUT"
    exit 2
}
[ ! -e "$LOG" ] || {
    echo "STOP: log already exists: $LOG"
    exit 2
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --repo "$REPO" \
    --g0-dir "$G0" \
    --data-root "$DATA" \
    --loader "$LOADER" \
    --model "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_g1a_status=$status"

if [ -f "$OUT/V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_COMPLETE" ]; then
    cat "$OUT/V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_COMPLETE"
elif [ -f "$OUT/V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD" ]; then
    cat "$OUT/V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
