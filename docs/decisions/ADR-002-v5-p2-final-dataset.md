# ADR-002: Treat V5 P2 as the Current Final Dataset

## Status

Accepted

## Context

Earlier V3 and V4 experiments exposed limitations involving dataset
coverage, chronological generalization, attacker localization, and
controlled architectural comparison.

V5 P2 introduced a pair-aligned window contract and a stricter
scientific protocol for source, graph, role, count, and multitask
prediction.

## Decision

V5 P2 is the authoritative dataset for all current model-selection,
validation, paper, and deployment claims.

V3 and V4 will remain in the repository as frozen historical
milestones. They must be clearly labelled as superseded and must not
be mixed with V5 P2 result tables.

## Consequences

- Current model configurations must reference V5 P2.
- G0 defines the frozen task and comparison protocol.
- The blind-test partition cannot influence model development.
- Any V5 P2 protocol amendment must be documented separately.
- Historical results must retain their original dataset labels.
