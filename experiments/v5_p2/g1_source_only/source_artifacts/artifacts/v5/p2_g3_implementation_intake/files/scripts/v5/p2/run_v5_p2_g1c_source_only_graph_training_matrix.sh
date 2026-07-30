#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/research/projects/GNN-2d}"
cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"

ONE_RUN="$REPO/scripts/v5/p2/run_v5_p2_g1c_source_only_graph_one_run.sh"
AGGREGATE="$REPO/scripts/v5/p2/run_v5_p2_g1c_source_only_graph_aggregation.sh"
PREFLIGHT_MARKER="$REPO/reports/v5/p2_g1c_clean_training_implementation_preflight/V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
LOCK_DIR="$REPO/reports/v5/.p2_g1c_source_only_graph_matrix.lock"

[ -f "$ONE_RUN" ] || { echo "STOP: missing one-run launcher: $ONE_RUN"; exit 2; }
[ -f "$AGGREGATE" ] || { echo "STOP: missing aggregation launcher: $AGGREGATE"; exit 2; }
[ -f "$PREFLIGHT_MARKER" ] || { echo "STOP: clean G1C implementation preflight is not complete"; exit 2; }

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "STOP: G1C matrix lock already exists: $LOCK_DIR"
    echo "Another launcher may be running, or a previous launch ended uncleanly."
    exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "===== V5 P2-G1C SERIAL 12-RUN MATRIX ====="
echo "order=seed-major"
echo "seeds=107,117,127"
echo "operators=conv1d,gcnconv,graphconv,gatconv"
echo "concurrent_gpu_processes=1"
echo "test_directory_enumerated=false"
echo "test_tensors_deserialized=false"
echo

for seed in 107 117 127; do
    for operator in conv1d gcnconv graphconv gatconv; do
        echo
        echo "===== MATRIX ITEM operator=$operator seed=$seed ====="
        bash "$ONE_RUN" "$operator" "$seed"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "HOLD: matrix stopped at operator=$operator seed=$seed status=$status"
            exit "$status"
        fi
    done
done

echo
echo "===== ALL 12 RUNS COMPLETE; AGGREGATING ====="
bash "$AGGREGATE"
status=$?
if [ "$status" -ne 0 ]; then
    echo "HOLD: all runs finished but aggregation failed"
    exit "$status"
fi

echo "V5_P2_G1C_SOURCE_ONLY_GRAPH_TRAINING_MATRIX_COMPLETE"
