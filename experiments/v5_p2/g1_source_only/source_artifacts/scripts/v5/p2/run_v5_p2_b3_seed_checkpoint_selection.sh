#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

B1="$REPO/reports/v5/p2_b1_training_protocol_lock"
B2="$REPO/reports/v5/p2_b2_multi_seed_training/finalization"
SEED_REPORT_ROOT="$REPO/reports/v5/p2_b2_multi_seed_training"
SEED_MODEL_ROOT="$REPO/models/v5/p2_b2_multi_seed"
LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL_SOURCE="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/select_v5_p2_b3_seed_checkpoint.py"
SELECTED_MODEL_DIR="$REPO/models/v5/p2_b3_selected_checkpoint"
OUT="$REPO/reports/v5/p2_b3_seed_checkpoint_selection"
LOG="$REPO/logs/v5/v5_p2_b3_seed_checkpoint_selection.log"

echo "===== V5 P2-B3 SEED + CHECKPOINT SELECTION ====="
echo "repo=$REPO"
echo "b1=$B1"
echo "b2=$B2"
echo "seed_report_root=$SEED_REPORT_ROOT"
echo "seed_model_root=$SEED_MODEL_ROOT"
echo "loader=$LOADER"
echo "model_source=$MODEL_SOURCE"
echo "script=$SCRIPT"
echo "selected_model_dir=$SELECTED_MODEL_DIR"
echo "output=$OUT"
echo "log=$LOG"
echo "selection_uses_validation_only=true"
echo "checkpoint_deserialized=false"
echo "threshold_tuning_performed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$B1" "$B2" "$SEED_REPORT_ROOT" "$SEED_MODEL_ROOT"; do
    [ -d "$path" ] || { echo "STOP: missing directory: $path"; exit 2; }
done
for path in "$LOADER" "$MODEL_SOURCE" "$SCRIPT"; do
    [ -f "$path" ] || { echo "STOP: missing file: $path"; exit 2; }
done
[ ! -e "$SELECTED_MODEL_DIR" ] || { echo "STOP: selected-model directory already exists: $SELECTED_MODEL_DIR"; exit 2; }
[ ! -e "$OUT" ] || { echo "STOP: output already exists: $OUT"; exit 2; }
[ ! -e "$LOG" ] || { echo "STOP: log already exists: $LOG"; exit 2; }

mkdir -p "$(dirname "$SELECTED_MODEL_DIR")" "$(dirname "$OUT")" "$(dirname "$LOG")"

python -u "$SCRIPT" \
  --b1-dir "$B1" \
  --b2-finalization-dir "$B2" \
  --seed-report-root "$SEED_REPORT_ROOT" \
  --seed-model-root "$SEED_MODEL_ROOT" \
  --loader-path "$LOADER" \
  --model-source-path "$MODEL_SOURCE" \
  --selected-model-dir "$SELECTED_MODEL_DIR" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
echo "v5_p2_b3_status=$status"

if [ -f "$OUT/V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_COMPLETE" ]; then
    cat "$OUT/V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_COMPLETE"
elif [ -f "$OUT/V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_HOLD" ]; then
    cat "$OUT/V5_P2_B3_SEED_AND_CHECKPOINT_SELECTION_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
