# V5 P0-C0 Final B3 Training Protocol

## Frozen architecture

- B3 causal depthwise-separable Conv1D-only
- 43,208 trainable parameters
- PRIMARY58, window 32, stride 8
- recovered topology-derived Boolean mask
- no graph message passing
- no metadata or provenance inputs

## Fresh seeds

`107, 117, 127`

These are fresh relative to the architecture-search seeds.

## Training

- train from scratch per seed
- maximum 150 epochs
- minimum 25 epochs
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- batch size 128
- gradient clipping at 5.0
- ReduceLROnPlateau, factor 0.5, patience 6
- early-stopping patience 20
- best-checkpoint restoration

## Model selection

Validation metrics are computed every epoch at fixed threshold 0.5.

```text
score =
  0.25 * graph balanced accuracy
+ 0.20 * graph F1
+ 0.10 * count macro F1
+ 0.10 * source F1
+ 0.10 * victim F1
+ 0.05 * transit F1
+ 0.05 * path F1
+ 0.15 * all-task exact
- 0.20 * max(0, 0.90 - graph recall)
- 0.10 * max(0, graph FPR - 0.20)
```

The best checkpoint from each seed is retained. One final checkpoint is selected
using validation only.

## Threshold calibration

C2 calibrates only the selected checkpoint on validation using the fixed grid:

`0.10, 0.15, ..., 0.90`

The resulting checkpoint and thresholds are hashed and frozen before test access.

## Test rule

C0, C1, and C2 must not construct or inspect the test split. C3 evaluates the
locked test exactly once.

## Protocol hash

`337fcd031b4ad4963c9a959a1c91075bda84dc913f35113e53077b26bfcd33e7`
