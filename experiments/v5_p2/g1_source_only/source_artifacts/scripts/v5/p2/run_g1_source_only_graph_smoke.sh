#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
R1="$REPO/reports/v5/p2_g1a_r1_source_only_graph_baseline_preflight"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/g1_source_only_graph_smoke.py"
OUT="$REPO/reports/v5/p2_g1_source_only_graph_smoke_check"
LOG="$REPO/logs/v5/v5_p2_g1_source_only_graph_smoke_check.log"

echo "===== G1 SOURCE-ONLY GRAPH SMOKE CHECK ====="
echo "repo=$REPO"
echo "g0=$G0"
echo "r1_manifest_resolution=$R1"
echo "topology=$TOPOLOGY"
echo "data=$DATA"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "historical_smoke_outputs_touched=false"
echo "architectures=conv1d,gcnconv,graphconv,gatconv"
echo "compact_balanced_selection=true"
echo "standard_and_pair_aligned_batches_supported=true"
echo "source_head_clone_equivalence_check=true"
echo "component_gradient_checks=true"
echo "identical_initialization_checks=true"
echo "emergency_failure_marker=true"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo "checkpoint_loaded=false"
echo "b4_validation_cache_accessed=false"
echo "b6_test_cache_accessed=false"
echo "full_training_performed=false"
echo "scientific_checkpoint_created=false"
echo "architecture_selected=false"
echo "quantization_performed=false"
echo "rtl_generated=false"
echo "legal_decoder_implemented=false"
echo

for path in "$G0" "$R1" "$TOPOLOGY" "$DATA"; do
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
    --r1-dir "$R1" \
    --topology-dir "$TOPOLOGY" \
    --data-root "$DATA" \
    --loader "$LOADER" \
    --model "$MODEL" \
    --output-dir "$OUT" \
    --batch-size 8 \
    --max-scan 1024 \
    --seed 107 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "g1_source_only_graph_smoke_status=$status"

if [ -f "$OUT/V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_COMPLETE" ]; then
    cat "$OUT/V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_COMPLETE"
elif [ -f "$OUT/V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_HOLD" ]; then
    cat "$OUT/V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
