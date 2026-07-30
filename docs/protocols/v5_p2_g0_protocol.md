# V5 P2 G0 Scientific Protocol

## Status

**FROZEN**

G0 defines the controlled experimental protocol used after the V5 P2
dataset contract was established.

## Tasks

### Task A — Source Localization

Predict which of the 16 routers contain malicious source IPs.

### Task B — Graph Detection

Predict whether the complete NoC traffic window represents normal
operation or a DDoS attack.

### Task C — Role-Aware Localization

Predict router roles, including malicious sources and victim nodes.

### Task D — Full Multitask Prediction

Jointly predict:

- graph attack status
- malicious source nodes
- attacker count
- victim nodes
- victim count

## Mandatory Operator Comparison

The following models must be compared under the same data, loss,
seed, optimizer, stopping, and evaluation protocol:

1. Conv1D temporal encoder only
2. Conv1D + GCNConv
3. Conv1D + GraphConv

GAT is retained only as a screening model and is not part of the
mandatory final comparison unless explicitly promoted by a later
protocol amendment.

## Model-Selection Policy

Architecture and checkpoint selection must use training and
validation information only.

GraphConv may be selected only from controlled comparisons produced
under this protocol. Its selection must not be described as a
universal result outside the evaluated dataset and task.

## Test Isolation

The blind-test partition remains inaccessible during:

- operator screening
- checkpoint selection
- threshold selection
- calibration
- protocol amendments

A single frozen evaluation command will be used for the final
blind-test assessment.

## Reproducibility Requirements

Every formal experiment must record:

- experiment identifier
- dataset version
- dataset-manifest checksum
- Git commit
- configuration file
- execution command
- environment
- random seed
- checkpoint checksum
- validation metrics
- test-access status
- completion status
