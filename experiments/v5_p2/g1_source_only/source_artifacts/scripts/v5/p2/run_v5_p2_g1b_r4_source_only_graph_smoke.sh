#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
R1="$REPO/reports/v5/p2_g1a_r1_source_only_graph_baseline_preflight"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
G1B_HOLD="$REPO/reports/v5/p2_g1b_source_only_graph_implementation_smoke"
G1B_R1_HOLD="$REPO/reports/v5/p2_g1b_r1_source_only_graph_implementation_smoke"
G1B_R2_HOLD="$REPO/reports/v5/p2_g1b_r2_source_only_graph_implementation_smoke"
DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/smoke_v5_p2_g1b_r4_source_only_graph_implementations.py"
OUT="$REPO/reports/v5/p2_g1b_r4_source_only_graph_implementation_smoke"
LOG="$REPO/logs/v5/v5_p2_g1b_r4_source_only_graph_implementation_smoke.log"

echo "===== V5 P2-G1B-R4 SOURCE-ONLY GRAPH SMOKE ====="
echo "repo=$REPO"
echo "g0=$G0"
echo "r1=$R1"
echo "topology=$TOPOLOGY"
echo "historical_g1b_hold=$G1B_HOLD"
echo "historical_g1b_r1_hold=$G1B_R1_HOLD"
echo "historical_g1b_r2_hold=$G1B_R2_HOLD"
echo "data=$DATA"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "historical_g1b_hold_preserved=true"
echo "historical_g1b_r1_hold_preserved=true"
echo "historical_g1b_r2_hold_preserved=true"
echo "architectures=conv1d,gcnconv,graphconv,gatconv"
echo "balanced_attack_control_smoke=true"
echo "pair_aligned_batch_flattening=true"
echo "exact_b3_source_head_clone=true"
echo "component_gradient_checks=true"
echo "identical_initialization_checks=true"
echo "full_b3_wrapper_production_use=false"
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

for path in \
    "$G0" \
    "$R1" \
    "$TOPOLOGY" \
    "$G1B_HOLD" \
    "$G1B_R1_HOLD" \
    "$G1B_R2_HOLD" \
    "$DATA"
do
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
    --g1b-hold-dir "$G1B_HOLD" \
    --g1b-r1-hold-dir "$G1B_R1_HOLD" \
    --g1b-r2-hold-dir "$G1B_R2_HOLD" \
    --data-root "$DATA" \
    --loader "$LOADER" \
    --model "$MODEL" \
    --output-dir "$OUT" \
    --batch-size 8 \
    --seed 107 \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_g1b_r4_status=$status"

if [ -f "$OUT/V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_COMPLETE" ]; then
    cat "$OUT/V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_COMPLETE"
elif [ -f "$OUT/V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD" ]; then
    cat "$OUT/V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
