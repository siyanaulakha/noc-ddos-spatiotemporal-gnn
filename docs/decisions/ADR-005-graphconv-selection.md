# ADR-005: Select GraphConv for the V5 P2 Spatial Operator

## Status

Accepted for the current V5 P2 protocol.

## Context

Earlier experiments used temporal and graph models under partially
different datasets and training conditions. These results were not
sufficient for a controlled operator conclusion.

The V5 P2 G0 protocol therefore required a comparison of:

1. Conv1D only
2. Conv1D with GCNConv
3. Conv1D with GraphConv

The comparison used the same source-localization task, data split,
loss policy, seed policy, checkpoint process, and validation metrics.

## Decision

Use GraphConv as the selected spatial operator for subsequent V5 P2
experiments unless a documented protocol amendment provides
contrary evidence.

## Basis

GraphConv produced the strongest validation behavior in the frozen
G1C comparison.

Reported highlights include:

- validation score of 0.7678 at epoch 36
- source AUC reaching 0.9867
- graph average precision reaching approximately 0.6967

These values must be verified against authoritative frozen metrics
files before being used in a publication table.

## Scope

This decision applies to:

- the V5 P2 dataset
- the frozen G1C source-localization task
- the evaluated Conv1D, GCNConv, and GraphConv models

It does not establish universal superiority over every spatial
operator or every Network-on-Chip dataset.

## Consequences

- G2, G3, and Task D should use GraphConv as the primary operator.
- Conv1D-only and GCNConv remain required baselines where the
  protocol calls for them.
- GAT remains a screening model unless promoted by a documented
  amendment.
- Final blind-test claims remain unavailable until the frozen
  one-shot test evaluation.
