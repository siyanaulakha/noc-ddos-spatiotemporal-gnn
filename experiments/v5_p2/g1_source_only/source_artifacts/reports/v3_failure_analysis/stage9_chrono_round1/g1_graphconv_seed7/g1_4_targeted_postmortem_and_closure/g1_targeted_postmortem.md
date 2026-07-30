# G1 Targeted Validation-Only Postmortem

## Formal closure

- G1 is rejected on validation.
- No formal test transfer is authorized.
- No additional G1 seed or V3 architecture experiment is authorized.
- Chrono-A1 remains the final V3 model.
- The separate-root insight is retained for a later V4-only normalized operator.

## Node behaviour, G1 minus A1

- `attacker_recall`: `+0.025711`
- `nonattacker_fpr_on_attack_samples`: `-0.002540`
- `exact_localization`: `+0.067112`
- `overprediction_rate`: `-0.027364`
- `underprediction_rate`: `-0.031110`
- `empty_prediction_rate`: `-0.003509`
- `adjacent_nonattacker_activation_rate`: `-0.009819`
- `other_nonattacker_activation_rate`: `+0.001099`
- `normal_node_activation_rate`: `+0.002078`

## Largest benign run regressions

- `N-2-6-10-14-Pmixed-R13-V3`: `+0.510780`
- `N-0-9-Pstream-R11-V3`: `+0.050106`
- `N-4-15-Pbursty-R12-V3`: `+0.013665`
- `N-idle-Pidle-R1-V3`: `+0.009110`

## Next architecture policy

1. Train V4-A1 with the normalized GCN first.
2. Perform V4-A1 run-level/localization failure analysis.
3. Test V4-G2 only if localization smearing remains.
4. G2 may separate root and neighbour weights, but the neighbour path must use mean or degree normalization.

Default-threshold V3 test metrics printed by the trainer are retained as diagnostic logs only.
