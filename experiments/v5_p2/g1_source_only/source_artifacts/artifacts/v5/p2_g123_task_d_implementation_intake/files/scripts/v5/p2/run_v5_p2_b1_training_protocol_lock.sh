#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

A1_R2="$REPO/reports/v5/p2_a1_r2_pair_aligned_window_contract"
A2_R2="$REPO/reports/v5/p2_a2_r2_feature_normalization_mask_contract"
A3="$REPO/reports/v5/p2_a3_pair_aligned_primary58_loader_contract"
B0_R2="$REPO/reports/v5/p2_b0_r2_count_head_expansion_a4_repeat"
B0_R3="$REPO/reports/v5/p2_b0_r3_corrected_nontest_label_shortcut_audit"

LOADER="$REPO/src/data/v5_p2_pair_aligned_primary58_dataset.py"
MODEL="$REPO/src/models/v5_p2_b3_conv1d_only_count4.py"
SCRIPT="$REPO/scripts/v5/p2/freeze_v5_p2_b1_training_protocol.py"

OUT="$REPO/reports/v5/p2_b1_training_protocol_lock"
LOG="$REPO/logs/v5/v5_p2_b1_training_protocol_lock.log"

echo "===== V5 P2-B1 TRAINING PROTOCOL LOCK ====="
echo "repo=$REPO"
echo "a1_r2=$A1_R2"
echo "a2_r2=$A2_R2"
echo "a3=$A3"
echo "b0_r2=$B0_R2"
echo "b0_r3=$B0_R3"
echo "loader=$LOADER"
echo "model=$MODEL"
echo "script=$SCRIPT"
echo "output=$OUT"
echo "log=$LOG"
echo "run_tensors_deserialized=false"
echo "training_performed=false"
echo "test_directory_enumerated=false"
echo "test_tensor_contents_accessed=false"
echo

for path in "$A1_R2" "$A2_R2" "$A3" "$B0_R2" "$B0_R3"; do
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
    --a1-r2-dir "$A1_R2" \
    --a2-r2-dir "$A2_R2" \
    --a3-dir "$A3" \
    --b0-r2-dir "$B0_R2" \
    --b0-r3-dir "$B0_R3" \
    --loader-path "$LOADER" \
    --model-path "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}

echo
echo "v5_p2_b1_status=$status"

if [ -f "$OUT/V5_P2_B1_TRAINING_PROTOCOL_LOCK_COMPLETE" ]; then
    cat "$OUT/V5_P2_B1_TRAINING_PROTOCOL_LOCK_COMPLETE"
elif [ -f "$OUT/V5_P2_B1_TRAINING_PROTOCOL_LOCK_HOLD" ]; then
    cat "$OUT/V5_P2_B1_TRAINING_PROTOCOL_LOCK_HOLD"
else
    echo "NO FINAL MARKER FOUND"
fi

exit "$status"
