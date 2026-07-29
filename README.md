# NoC DDoS Spatiotemporal GNN

Topology-aware spatiotemporal machine-learning framework for
detection and localization of distributed denial-of-service attacks
in Network-on-Chip systems.

## Current Status

- Current dataset: V5 P2
- Current frozen protocol: V5 P2 G0
- Completed: G1C source-only operator comparison
- Planned: G2, G3, Task D, final blind-test evaluation, RTL handoff

## Important

V3 and V4 are preserved as historical and superseded development
stages. Current architectural and experimental claims must use the
V5 P2 dataset and frozen protocol.

## Repository Structure

- `configs/` — dataset, model, task, and protocol configurations
- `src/` — reusable Python implementation
- `scripts/` — dataset, training, evaluation, audit, and export scripts
- `experiments/` — reproducible experiment records
- `reports/` — tables, figures, audits, and failure analysis
- `docs/` — project history, architecture, protocols, and decisions
- `rtl/` — future quantization and RTL handoff material
- `tests/` — correctness and reproducibility tests

## Project Tasks

- Task A — source-node localization
- Task B — graph-level attack detection
- Task C — role-aware attacker and victim localization
- Task D — complete multitask prediction

## Model Comparisons

The frozen operator comparison includes:

- Conv1D only
- Conv1D + GCNConv
- Conv1D + GraphConv

GAT is retained only as a screening model.

## Reproducibility

Every formal experiment should record:

- dataset manifest and checksum
- Git commit
- configuration
- random seed
- training command
- environment
- checkpoint checksum
- validation metrics
- test-access status
