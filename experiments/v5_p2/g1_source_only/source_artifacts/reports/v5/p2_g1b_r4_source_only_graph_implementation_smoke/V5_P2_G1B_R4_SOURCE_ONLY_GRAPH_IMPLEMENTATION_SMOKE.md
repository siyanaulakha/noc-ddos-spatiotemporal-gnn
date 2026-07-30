# V5 P2-G1B-R4 Source-Only Graph Implementation Smoke

- Status: **HOLD**
- Device: `None`
- Batch size: `8`

This stage performs exactly one smoke optimizer step per architecture. It does not conduct the scientific three-seed training matrix or select an architecture.

Architectures:

- Conv1D-only
- Conv1D + GCNConv
- Conv1D + GraphConv
- Conv1D + GATConv

The next stage may launch the serial 12-run matrix only after building a clean source-only model that extracts the temporal encoder and removes all unused B3 task heads. The hook-based full B3 wrapper in this smoke is an interface oracle, not the production training architecture.
