# V5 P2 Experiments

V5 P2 is the current active experimental track.

| Stage | Purpose | Status |
|---|---|---|
| G0 | Dataset, task, comparison, and test-isolation protocol | FROZEN |
| G1C | Source-only temporal/spatial operator comparison | COMPLETED |
| G2 | Direct graph-level attack detection | PLANNED |
| G3 | Role-aware attacker and victim localization | PLANNED |
| Task D | Complete multitask model | PLANNED |
| Blind test | Frozen one-shot final evaluation | PLANNED |

## Completed G1C Comparison

The controlled operator screening included:

- Conv1D only — 28,353 parameters
- Conv1D + GCNConv — 36,673 parameters
- Conv1D + GraphConv — 44,865 parameters

Primary screening seed: 107.

GraphConv produced the strongest validation performance in the
completed controlled screening. This is a validation-stage
architecture-selection result, not yet the final blind-test claim.

## Experiment Directory Requirements

Every completed experiment directory must contain:

- `README.md`
- `status.yaml`
- `config.yaml`
- `command.sh`
- `environment.json`
- `metrics.json`
- `history.csv`
- `checkpoint.sha256`
- dataset-manifest reference
- test-access audit
