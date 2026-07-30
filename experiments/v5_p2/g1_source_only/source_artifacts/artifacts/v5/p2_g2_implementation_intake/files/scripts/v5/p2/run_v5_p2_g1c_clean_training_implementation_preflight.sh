#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

export PYTHONHASHSEED=107
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA="$REPO/data/processed/v5/p2_complete558_dynamic_graph"
G0="$REPO/reports/v5/p2_g0_graph_baseline_rtl_handoff_protocol"
SMOKE="$REPO/reports/v5/p2_g1_source_only_graph_smoke_check"
TOPOLOGY="$REPO/reports/v5/p2_g1a_r2a_canonical_static_topology_contract"
B0_R3="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"
PAIR_MANIFEST="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
B3_MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
TRAINER="$REPO/scripts/v5/p2/train_v5_p2_g1c_source_only_graph_single_run.py"
SCRIPT="$REPO/scripts/v5/p2/preflight_v5_p2_g1c_clean_training_implementation.py"
OUT="$REPO/reports/v5/p2_g1c_clean_training_implementation_preflight"
LOG="$REPO/logs/v5/v5_p2_g1c_clean_training_implementation_preflight.log"
COMPLETE="$OUT/V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
HOLD="$OUT/V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_HOLD"

if [ -f "$COMPLETE" ]; then
    echo "G1C clean implementation preflight already complete"
    cat "$COMPLETE"
    exit 0
fi

for path in "$DATA" "$G0" "$SMOKE" "$TOPOLOGY" "$B0_R3"; do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done
for path in "$PAIR_MANIFEST" "$LOADER" "$B3_MODEL" "$TRAINER" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done
if [ -e "$OUT" ] || [ -e "$LOG" ]; then
    echo "STOP: partial clean-preflight output exists"
    echo "out=$OUT"
    echo "log=$LOG"
    [ -f "$HOLD" ] && cat "$HOLD"
    exit 2
fi
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
    --root "$DATA" \
    --g0-dir "$G0" \
    --smoke-dir "$SMOKE" \
    --topology-dir "$TOPOLOGY" \
    --b0-r3-dir "$B0_R3" \
    --pair-manifest "$PAIR_MANIFEST" \
    --loader-path "$LOADER" \
    --b3-model-path "$B3_MODEL" \
    --trainer-path "$TRAINER" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "v5_p2_g1c_clean_preflight_status=$status"
[ -f "$COMPLETE" ] && cat "$COMPLETE"
[ -f "$HOLD" ] && cat "$HOLD"
exit "$status"
