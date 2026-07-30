# V5 P2-G1C Source-Only Graph Training Package

This package implements the controlled G1C experiment:

- Conv1D-only
- Conv1D + GCNConv
- Conv1D + GraphConv
- Conv1D + GATConv screening
- seeds 107, 117, and 127
- serial execution: one GPU process at a time
- P2 train and validation only
- no P2 test enumeration, deserialization, or inference

## Required execution order

1. Install all package files into `scripts/v5/p2/`.
2. Run the clean-production implementation preflight.
3. Confirm `V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE`.
4. Launch the serial 12-run matrix.
5. The matrix automatically runs aggregation after all twelve runs complete.

## Why another preflight exists

The completed G1 smoke used a hook-based wrapper around the full B3 model to
prove the operator concepts. The frozen smoke contract explicitly disallows
that structure in production. This package therefore extracts only:

- B3 input projection;
- four causal depthwise-separable temporal blocks;
- node projection;
- source head.

No graph/count/transit/victim/path B3 heads are retained. The clean preflight
proves zero-error source-logit equivalence against the seeded full B3 reference
before authorizing scientific training.

## Clean production parameter counts

- Conv1D-only: 28,353
- GCNConv: 36,673
- GraphConv: 44,865
- GATConv: 36,929

These are intentionally lower than the earlier smoke-wrapper counts because
unused multitask heads are removed.

## Frozen training settings

- item batch size: 256, represented as 128 aligned ATTACK/CONTROL blocks;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0001;
- scheduler: ReduceLROnPlateau, factor 0.5, patience 4, min LR 1e-5;
- maximum epochs: 100;
- minimum epoch before stopping: 15;
- early-stop patience: 12;
- early-stop minimum delta: 1e-4;
- global gradient clipping: 1.0;
- AMP: disabled;
- source loss: class-weighted BCEWithLogitsLoss.

Checkpoint score:

```text
0.50 * source AP
+ 0.25 * source AUROC
+ 0.25 * derived-graph AP
```

Post-checkpoint source-threshold objective:

```text
0.60 * source-node F1
+ 0.40 * exact attacker-set accuracy on attack samples
```

The G0 documents freeze the objective but not a numerical threshold grid. This
package makes that previously implicit implementation detail explicit and
common to every run: thresholds 0.000 through 1.000 inclusive at step 0.001.
Threshold tuning occurs only after the best checkpoint is selected.

## Output locations

Per-run models:

```text
models/v5/p2_g1c_source_only_graph/<operator>/seed_<seed>/
```

Per-run reports:

```text
reports/v5/p2_g1c_source_only_graph_training/<operator>/seed_<seed>/
```

Aggregation:

```text
reports/v5/p2_g1c_source_only_graph_matrix_aggregation/
```

The aggregation reports per-seed values, mean and sample standard deviation,
parameter count, analytic operation proxy, peak CUDA memory, inference latency,
and the explicit GAT-vs-GraphConv promotion gates. It does not select the final
RTL architecture.
