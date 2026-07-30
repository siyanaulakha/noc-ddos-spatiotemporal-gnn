# V5 P2 B1 Training Protocol Lock

## Corrected architecture

```text
model       = B3_CAUSAL_DEPTHWISE_SEPARABLE_CONV1D_ONLY_P2_COUNT4
parameters  = 43273
count head  = 4 active-count logits
classes     = 1, 2, 3, 4
```

Only the P0 count head changed from `Linear(64,3)` to `Linear(64,4)`.
The temporal encoder, graph detector, and four localization heads remain
unchanged.

## Dataset

```text
train items       = 70166
validation items  = 12528
window / stride   = 32 / 8
input             = [B,16,58,32]
mask              = [B,16,10]
batch             = 256 items = 128 aligned pair-windows
```

Aligned ATTACK/CONTROL blocks are shuffled as units. The order inside each
two-item block remains ATTACK then CONTROL.

## Runs

```text
seeds       = [107, 117, 127]
max epochs  = 100
early stop  = patience 12 after minimum epoch 15
precision   = float32
AMP         = disabled
```

## Optimizer

```text
AdamW
lr           = 1e-3
weight decay = 1e-4
gradient clip global norm = 1.0
```

`ReduceLROnPlateau` monitors the validation selection score with factor 0.5,
patience 4, and minimum learning rate `1e-5`.

## Checkpoint-selection score

```text
0.30 graph AUROC
0.15 graph average precision
0.15 active count macro F1
0.10 source average precision
0.10 transit average precision
0.10 victim average precision
0.10 path average precision
```

No threshold tuning occurs during B2. The score is threshold-free except for
count-head argmax.

## Loss

```text
1.00 attack BCE
0.50 active-only four-class count CE
1.00 source BCE
0.50 transit BCE
0.75 victim BCE
0.50 path BCE
```

`role_mask` is the bit-field `source + 2*transit + 4*victim`. It is
bookkeeping only and receives no independent loss.

## Test boundary

P2 test remains unauthorized. It may be evaluated once only after multi-seed
training, seed selection, checkpoint freeze, and validation-only threshold
freeze.

## Protocol SHA-256

`0817ae7812f91e3c75260589acaf52bd8a74b07b2114a451b4773578efde4b60`
