#!/usr/bin/env bash
set -uo pipefail

OPERATOR="${1:-}"
SEED="${2:-}"

case "$OPERATOR" in
    conv1d|gcnconv|graphconv|gatconv) ;;
    *)
        echo "Usage: $0 {conv1d|gcnconv|graphconv|gatconv} {107|117|127}"
        exit 2
        ;;
esac
case "$SEED" in
    107|117|127) ;;
    *)
        echo "Usage: $0 {conv1d|gcnconv|graphconv|gatconv} {107|117|127}"
        exit 2
        ;;
esac

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
SMOKE="$REPO/reports/v5/p2_g1_source_only_graph_smoke_check"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
B0_R3="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"
PREFLIGHT="$REPO/reports/v5/p2_g1c_clean_training_implementation_preflight"
PAIR_MANIFEST="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
B3_MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/train_v5_p2_g1c_source_only_graph_single_run.py"

MODEL_DIR="$REPO/models/v5/p2_g1c_source_only_graph/$OPERATOR/seed_${SEED}"
REPORT_DIR="$REPO/reports/v5/p2_g1c_source_only_graph_training/$OPERATOR/seed_${SEED}"
LOG="$REPO/logs/v5/v5_p2_g1c_${OPERATOR}_seed_${SEED}.log"

COMPLETE_MARKER="$REPORT_DIR/V5_P2_G1C_SOURCE_ONLY_GRAPH_SINGLE_RUN_COMPLETE"
HOLD_MARKER="$REPORT_DIR/V5_P2_G1C_SOURCE_ONLY_GRAPH_SINGLE_RUN_HOLD"

if [ -f "$COMPLETE_MARKER" ]; then
    echo "ALREADY COMPLETE: $OPERATOR seed $SEED"
    cat "$COMPLETE_MARKER"
    exit 0
fi

for path in "$DATA" "$G0" "$SMOKE" "$TOPOLOGY" "$B0_R3" "$PREFLIGHT"; do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done
for path in "$PAIR_MANIFEST" "$LOADER" "$B3_MODEL" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done

if [ -e "$MODEL_DIR" ] || [ -e "$REPORT_DIR" ] || [ -e "$LOG" ]; then
    echo "STOP: partial or pre-existing output exists for $OPERATOR seed $SEED"
    echo "model_dir=$MODEL_DIR"
    echo "report_dir=$REPORT_DIR"
    echo "log=$LOG"
    [ -f "$HOLD_MARKER" ] && cat "$HOLD_MARKER"
    exit 2
fi

mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$REPORT_DIR")" "$(dirname "$LOG")"

echo "===== V5 P2-G1C SOURCE-ONLY GRAPH RUN ====="
echo "operator=$OPERATOR"
echo "seed=$SEED"
echo "execution=serial_one_model_at_a_time"
echo "automatic_mixed_precision=false"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo "architecture_selected=false"
echo

python -u "$SCRIPT" \
    --root "$DATA" \
    --g0-dir "$G0" \
    --smoke-dir "$SMOKE" \
    --topology-dir "$TOPOLOGY" \
    --b0-r3-dir "$B0_R3" \
    --pair-manifest "$PAIR_MANIFEST" \
    --preflight-dir "$PREFLIGHT" \
    --loader-path "$LOADER" \
    --b3-model-path "$B3_MODEL" \
    --operator "$OPERATOR" \
    --seed "$SEED" \
    --model-dir "$MODEL_DIR" \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
echo "v5_p2_g1c_${OPERATOR}_seed_${SEED}_status=$status"

if [ -f "$COMPLETE_MARKER" ]; then
    cat "$COMPLETE_MARKER"
elif [ -f "$HOLD_MARKER" ]; then
    cat "$HOLD_MARKER"
else
    echo "NO FINAL MARKER FOUND"
fi
exit "$status"
