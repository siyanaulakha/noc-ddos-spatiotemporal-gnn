# V5 P2 Dataset Card

## Status

**ACTIVE — current authoritative project dataset**

V3 and V4 remain preserved for historical reproducibility but must
not be used for current architectural or final-performance claims.

## Purpose

V5 P2 supports controlled experiments for:

- graph-level DDoS detection
- malicious-source localization
- role-aware attacker and victim localization
- attacker-count prediction
- victim-count prediction
- full multitask inference

## Dataset Contract

The dataset uses pair-aligned spatiotemporal windows so that traffic
features and corresponding source, victim, role, graph, and count
labels refer to the same simulation interval.

The authoritative sample mapping is stored in:

`data/manifests/v5_p2_pair_aligned_window_manifest.csv`

## Split Policy

Training, validation, and blind-test partitions must remain separated
at the simulation-run and scenario level.

Overlapping temporal windows must never cross split boundaries.

## Blind-Test Policy

The blind-test set must not be used for:

- architecture selection
- checkpoint selection
- threshold selection
- early stopping
- loss-weight selection
- feature selection
- hyperparameter tuning

Before the final one-shot evaluation:

- test directories must not be enumerated by training code
- test tensors must not be deserialized
- test metrics must not be computed

## Current Training Context

For the completed G1C source-only operator comparison:

- training items: 70,166
- validation items: 12,528
- training batches: 275
- validation batches: 49
- source positive-class weight: 20.0
- primary screening seed: 107

These values describe the G1C protocol and are not a complete
description of the entire V5 P2 dataset.

## Known Limitations

The complete final limitations section will be frozen after G2, G3,
Task D, and the one-shot blind-test evaluation are complete.
