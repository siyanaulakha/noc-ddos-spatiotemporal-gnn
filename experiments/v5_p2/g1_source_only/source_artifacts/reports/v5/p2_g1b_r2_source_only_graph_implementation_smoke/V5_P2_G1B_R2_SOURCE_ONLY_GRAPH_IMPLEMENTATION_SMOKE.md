# V5 P2-G1B-R2 Source-Only Graph Implementation Smoke

- Status: **HOLD**
- Device: `None`
- Batch size: `8`

This stage performs exactly one smoke optimizer step per architecture. It does not conduct the scientific three-seed training matrix or select an architecture.

Architectures:

- Conv1D-only
- Conv1D + GCNConv
- Conv1D + GraphConv
- Conv1D + GATConv

The next stage launches the serial 12-run matrix only after every architecture produces finite logits, losses, and gradients with the same B3 node-embedding interface.
