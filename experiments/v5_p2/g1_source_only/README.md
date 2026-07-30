# V5 P2 G1C — Source-Only Operator Comparison

## Status

**COMPLETED — validation-stage architecture screening**

G1C compares temporal-only and spatiotemporal models under one
frozen V5 P2 protocol.

## Task

The model predicts which routers contain malicious source IPs.

This is Task A of the V5 P2 protocol.

## Controlled Models

| Model | Spatial operator | Parameters |
|---|---:|---:|
| Conv1D | None | 28,353 |
| Conv1D + GCNConv | GCNConv | 36,673 |
| Conv1D + GraphConv | GraphConv | 44,865 |

All operators must be evaluated using the same:

- dataset split
- training items
- validation items
- optimizer policy
- loss definition
- positive-class weight
- checkpoint policy
- stopping criteria
- random-seed policy
- evaluation metrics

## Frozen Training Context

- Primary screening seed: 107
- Training items: 70,166
- Validation items: 12,528
- Training batches: 275
- Validation batches: 49
- Source positive-class weight: 20.0

## Selection

GraphConv was selected from the controlled validation comparison.

The selection means:

> GraphConv was the strongest evaluated spatial operator for the
> frozen V5 P2 G1C source-localization protocol.

It does not mean that GraphConv is universally superior for all
datasets, topologies, tasks, or hardware targets.

## Test Isolation

During G1C model selection:

- the blind-test directory was not enumerated
- blind-test tensors were not deserialized
- blind-test metrics were not computed
- blind-test results did not influence architecture selection

## Directory Contents

- `status.yaml` — formal experiment status
- `reported_results.yaml` — currently confirmed result summary
- `source_manifest.tsv` — source-file provenance and checksums
- `g1c_source_candidates.txt` — discovered candidate artefacts
- `source_artifacts/` — copied lightweight protocol and result files
- `checksums/` — checksums for curated repository artefacts

## Remaining Work

- derive one authoritative per-operator metrics table
- connect every selected checkpoint to its checksum
- record the exact Git commit used during training
- preserve the exact runtime environment
- validate all reconstructed values against source result files
- keep the blind test inaccessible until final evaluation
