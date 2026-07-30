#!/usr/bin/env python3
"""
G0: UofF-cited GraphConv versus Chrono-A1 GCN equivalence audit.

This stage performs:
- no model training;
- no validation/test inference;
- no source modification.

It verifies the frozen A1 source, the V3 topology/self-loop convention,
the exact algebraic difference between A1 GCN and the Morris/PyG GraphConv
operator, parameter counts, and reference forward equivalence.

Important:
V3 edge_index includes self-loops. PyG GraphConv already has a separate
root/self transform, so self-loops must be removed from the neighbor edge set
to avoid double-counting the root node.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_A1_PARAMETER_COUNT = 882
EXPECTED_G1_PARAMETER_COUNT = 1138
EXPECTED_PARAMETER_DELTA = 256


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DenseGraphConv(nn.Module):
    """Dense exact equivalent of PyG GraphConv with add aggregation.

    x'_i = W_root x_i + W_neighbor sum_{j in N(i)} x_j + b_neighbor

    The supplied adjacency must exclude self-loops.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_neighbor = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_root = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        neighbor_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_sum = torch.einsum(
            "ij,bjf->bif",
            neighbor_adjacency,
            h,
        )
        return self.lin_neighbor(neighbor_sum) + self.lin_root(h)


class TemporalGraphConv(nn.Module):
    """A1-compatible reference model changing only the graph operator."""

    def __init__(
        self,
        temporal_module: nn.Module,
        temporal_dim: int = 8,
        graph_hidden: int = 16,
        graph_out: int = 8,
    ):
        super().__init__()
        self.temporal = temporal_module
        self.graph1 = DenseGraphConv(temporal_dim, graph_hidden)
        self.graph2 = DenseGraphConv(graph_hidden, graph_out)
        self.node_head = nn.Linear(graph_out, 1)
        self.graph_head = nn.Linear(graph_out, 1)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.temporal(x)
        h = torch.relu(self.graph1(h, neighbor_adjacency))
        h = torch.relu(self.graph2(h, neighbor_adjacency))
        node_logits = self.node_head(h).squeeze(-1)
        graph_logits = self.graph_head(h.mean(dim=1)).squeeze(-1)
        return graph_logits, node_logits


def topology_report(edge_index: np.ndarray, num_nodes: int) -> dict[str, Any]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")

    edges = [
        (int(source), int(target))
        for source, target in zip(edge_index[0], edge_index[1])
    ]
    unique_edges = set(edges)
    self_edges = sorted(edge for edge in unique_edges if edge[0] == edge[1])
    nonself_edges = sorted(edge for edge in unique_edges if edge[0] != edge[1])
    missing_reverse = sorted(
        edge for edge in nonself_edges
        if (edge[1], edge[0]) not in unique_edges
    )

    expected_self_edges = [(node, node) for node in range(num_nodes)]

    neighbor_adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float32,
    )
    for source, target in nonself_edges:
        neighbor_adjacency[target, source] = 1.0

    degrees = neighbor_adjacency.sum(dim=1).to(torch.int64).tolist()

    return {
        "edge_index_shape": list(edge_index.shape),
        "edge_count_raw": int(edge_index.shape[1]),
        "edge_count_unique": len(unique_edges),
        "duplicate_edge_count": int(edge_index.shape[1] - len(unique_edges)),
        "self_loop_count": len(self_edges),
        "self_loops": [list(edge) for edge in self_edges],
        "all_expected_self_loops_present": self_edges == expected_self_edges,
        "nonself_directed_edge_count": len(nonself_edges),
        "missing_reverse_edges": [list(edge) for edge in missing_reverse],
        "bidirectional_nonself_edges": len(missing_reverse) == 0,
        "neighbor_degrees_without_self": degrees,
        "neighbor_adjacency": neighbor_adjacency,
    }


def graph_parameter_counts() -> dict[str, int]:
    # A1: one affine transform per layer.
    a1_layer1 = 8 * 16 + 16
    a1_layer2 = 16 * 8 + 8
    a1_graph_block = a1_layer1 + a1_layer2

    # GraphConv: biased neighbor transform + bias-free root transform.
    g1_layer1 = (8 * 16 + 16) + (8 * 16)
    g1_layer2 = (16 * 8 + 8) + (16 * 8)
    g1_graph_block = g1_layer1 + g1_layer2

    return {
        "a1_graph_layer1": a1_layer1,
        "a1_graph_layer2": a1_layer2,
        "a1_graph_block": a1_graph_block,
        "g1_graph_layer1": g1_layer1,
        "g1_graph_layer2": g1_layer2,
        "g1_graph_block": g1_graph_block,
        "graph_block_delta": g1_graph_block - a1_graph_block,
    }


def operation_estimates(
    *,
    num_nodes: int,
    nonself_edges: int,
    edge_count_with_self: int,
) -> dict[str, Any]:
    """Approximate scalar multiply/accumulate work for two graph layers.

    Sparse estimates count one feature accumulation per directed edge and
    dense feature transforms. Dense estimates match the current A1 einsum
    implementation style.
    """
    dims = [(8, 16), (16, 8)]

    a1_sparse = 0
    a1_dense = 0
    g1_sparse = 0
    g1_dense = 0

    layer_rows = []
    for layer_index, (fin, fout) in enumerate(dims, start=1):
        a1_sparse_layer = edge_count_with_self * fin + num_nodes * fin * fout
        a1_dense_layer = num_nodes * num_nodes * fin + num_nodes * fin * fout

        g1_sparse_layer = (
            nonself_edges * fin
            + num_nodes * fin * fout
            + num_nodes * fin * fout
        )
        g1_dense_layer = (
            num_nodes * num_nodes * fin
            + num_nodes * fin * fout
            + num_nodes * fin * fout
        )

        a1_sparse += a1_sparse_layer
        a1_dense += a1_dense_layer
        g1_sparse += g1_sparse_layer
        g1_dense += g1_dense_layer

        layer_rows.append(
            {
                "layer": layer_index,
                "in_dim": fin,
                "out_dim": fout,
                "a1_sparse_estimate": a1_sparse_layer,
                "a1_dense_estimate": a1_dense_layer,
                "g1_sparse_estimate": g1_sparse_layer,
                "g1_dense_estimate": g1_dense_layer,
            }
        )

    return {
        "definition": (
            "Approximate scalar feature accumulations plus linear-layer "
            "multiply-accumulates; excludes activation, bias, launch, and "
            "framework overhead."
        ),
        "layers": layer_rows,
        "a1_sparse_total": a1_sparse,
        "g1_sparse_total": g1_sparse,
        "g1_vs_a1_sparse_ratio": g1_sparse / a1_sparse,
        "a1_dense_total": a1_dense,
        "g1_dense_total": g1_dense,
        "g1_vs_a1_dense_ratio": g1_dense / a1_dense,
    }


def manual_graphconv(
    layer: DenseGraphConv,
    h: torch.Tensor,
    neighbor_adjacency: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_nodes, _ = h.shape
    output = torch.empty(
        batch_size,
        num_nodes,
        layer.lin_neighbor.out_features,
        dtype=h.dtype,
    )

    neighbor_weight = layer.lin_neighbor.weight
    neighbor_bias = layer.lin_neighbor.bias
    root_weight = layer.lin_root.weight

    for batch in range(batch_size):
        for target in range(num_nodes):
            neighbor_sum = torch.zeros(
                h.shape[-1],
                dtype=h.dtype,
            )
            for source in range(num_nodes):
                if neighbor_adjacency[target, source] != 0:
                    neighbor_sum += h[batch, source]

            output[batch, target] = (
                neighbor_sum @ neighbor_weight.T
                + neighbor_bias
                + h[batch, target] @ root_weight.T
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    a1_source = args.a1_source.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not a1_source.is_file():
        raise SystemExit(f"STOP: A1 source missing: {a1_source}")
    if not data_dir.is_dir():
        raise SystemExit(f"STOP: data directory missing: {data_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: G0 output directory already non-empty: {output_dir}"
        )

    x_path = data_dir / "x.npy"
    edge_path = data_dir / "edge_index.npy"
    if not x_path.is_file() or not edge_path.is_file():
        raise SystemExit("STOP: x.npy or edge_index.npy missing.")

    source_hash = sha256(a1_source)
    module = load_module(a1_source, "chrono_a1_g0_audit")

    x = np.load(x_path, mmap_mode="r")
    edge_index = np.load(edge_path).astype(np.int64)
    num_nodes = int(x.shape[1])
    input_features = int(x.shape[-1])

    topology = topology_report(edge_index, num_nodes)
    neighbor_adjacency = topology.pop("neighbor_adjacency")

    torch.manual_seed(20260721)
    a1_model = module.TemporalGCN(
        input_features=input_features,
        temporal_dim=8,
        gcn_hidden=16,
        gcn_out=8,
    )
    a1_parameters = sum(parameter.numel() for parameter in a1_model.parameters())

    torch.manual_seed(20260721)
    temporal_for_g1 = module.TemporalEncoder(
        input_features,
        8,
        kernel_size=3,
    )
    g1_model = TemporalGraphConv(
        temporal_module=temporal_for_g1,
        temporal_dim=8,
        graph_hidden=16,
        graph_out=8,
    )
    g1_parameters = sum(parameter.numel() for parameter in g1_model.parameters())

    count_formula = graph_parameter_counts()
    operation_estimate = operation_estimates(
        num_nodes=num_nodes,
        nonself_edges=topology["nonself_directed_edge_count"],
        edge_count_with_self=topology["edge_count_unique"],
    )

    torch.manual_seed(7301)
    reference_layer = DenseGraphConv(8, 16)
    random_h = torch.randn(3, num_nodes, 8)
    direct = reference_layer(random_h, neighbor_adjacency)
    manual = manual_graphconv(
        reference_layer,
        random_h,
        neighbor_adjacency,
    )
    formula_max_abs_error = float(torch.max(torch.abs(direct - manual)).item())

    # Demonstrate why keeping self-loops in the neighbor graph is wrong:
    adjacency_with_self = neighbor_adjacency + torch.eye(num_nodes)
    incorrectly_double_counted = reference_layer(
        random_h,
        adjacency_with_self,
    )
    double_count_difference = float(
        torch.max(torch.abs(direct - incorrectly_double_counted)).item()
    )

    pyg_info: dict[str, Any]
    try:
        import torch_geometric  # type: ignore
        from torch_geometric.nn import GraphConv  # type: ignore

        pyg_layer = GraphConv(8, 16, aggr="add", bias=True)
        pyg_parameters = sum(
            parameter.numel() for parameter in pyg_layer.parameters()
        )
        pyg_info = {
            "available": True,
            "version": str(torch_geometric.__version__),
            "single_layer_8_to_16_parameter_count": pyg_parameters,
            "expected_single_layer_parameter_count": 272,
            "parameter_count_matches_reference": pyg_parameters == 272,
        }
    except Exception as error:
        pyg_info = {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
            "required_for_equation_audit": False,
            "note": (
                "The dense reference implements the documented GraphConv "
                "equation exactly; PyG is not required for this audit."
            ),
        }

    checks = {
        "a1_source_hash_matches_frozen": (
            source_hash == EXPECTED_A1_SOURCE_SHA256
        ),
        "dataset_shape_is_four_dimensional": len(x.shape) == 4,
        "node_count_is_16": num_nodes == 16,
        "edge_index_shape_is_2_by_64": list(edge_index.shape) == [2, 64],
        "all_16_self_loops_present": topology["all_expected_self_loops_present"],
        "nonself_edges_are_bidirectional": topology["bidirectional_nonself_edges"],
        "nonself_directed_edge_count_is_48": (
            topology["nonself_directed_edge_count"] == 48
        ),
        "a1_parameter_count_is_882": (
            a1_parameters == EXPECTED_A1_PARAMETER_COUNT
        ),
        "g1_parameter_count_is_1138": (
            g1_parameters == EXPECTED_G1_PARAMETER_COUNT
        ),
        "parameter_delta_is_256": (
            g1_parameters - a1_parameters == EXPECTED_PARAMETER_DELTA
        ),
        "graphconv_formula_matches_manual_reference": (
            formula_max_abs_error <= 1e-6
        ),
        "self_loop_double_counting_is_observable": (
            double_count_difference > 1e-6
        ),
    }

    status = (
        "EQUIVALENCE_AUDIT_PASSED"
        if all(checks.values())
        else "EQUIVALENCE_AUDIT_FAILED"
    )

    report = {
        "stage": "G0",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "training_started": False,
            "validation_inference_performed": False,
            "test_inference_performed": False,
            "source_modified": False,
        },
        "frozen_a1": {
            "source": str(a1_source),
            "source_sha256": source_hash,
            "parameter_count": a1_parameters,
            "operator": (
                "D^{-1/2} A D^{-1/2} H W + b, where V3 A already "
                "contains self-loops"
            ),
        },
        "uof_cited_graphconv": {
            "reference": (
                "Morris et al./PyTorch Geometric GraphConv formulation"
            ),
            "equation": (
                "H'_i = W_root H_i + W_neighbor "
                "sum_{j in N(i)} H_j + b_neighbor"
            ),
            "aggregation": "add",
            "separate_root_and_neighbor_transforms": True,
            "degree_normalization": False,
            "self_loop_policy": (
                "Remove self-loops from neighbor edges; root path handles self"
            ),
            "exact_uof_implementation_available": False,
            "interpretation": (
                "This is an operator-faithful ablation based on the paper's "
                "cited GraphConv, not a reproduction of the complete UofF model."
            ),
        },
        "topology": topology,
        "parameters": {
            "a1_total": a1_parameters,
            "g1_total": g1_parameters,
            "delta": g1_parameters - a1_parameters,
            "percentage_increase": (
                100.0 * (g1_parameters - a1_parameters) / a1_parameters
            ),
            "graph_layer_formula_counts": count_formula,
        },
        "operation_estimate": operation_estimate,
        "reference_forward": {
            "formula_max_abs_error": formula_max_abs_error,
            "incorrect_self_loop_double_count_max_abs_difference": (
                double_count_difference
            ),
        },
        "pytorch_geometric": pyg_info,
        "checks": checks,
        "decision_if_passed": {
            "operator_meaningfully_different": True,
            "bounded_v3_diagnostic_justified": True,
            "broad_v3_search_reopened": False,
            "final_v3_model_changed": False,
            "next_authorized_stage": (
                "G1 source generation and structural diff audit only"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "graphconv_equivalence_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("G0 UOF GRAPHCONV EQUIVALENCE AUDIT")
    print(f"status={status}")
    print(f"a1_source_sha256={source_hash}")
    print(f"input_shape={tuple(x.shape)}")
    print(f"edge_index_shape={tuple(edge_index.shape)}")
    print(f"self_loop_count={topology['self_loop_count']}")
    print(
        "nonself_directed_edge_count="
        f"{topology['nonself_directed_edge_count']}"
    )
    print(
        "bidirectional_nonself_edges="
        f"{topology['bidirectional_nonself_edges']}"
    )
    print(f"a1_parameter_count={a1_parameters}")
    print(f"g1_parameter_count={g1_parameters}")
    print(f"parameter_delta={g1_parameters - a1_parameters}")
    print(
        "parameter_increase_percent="
        f"{report['parameters']['percentage_increase']:.3f}"
    )
    print(
        "graphconv_formula_max_abs_error="
        f"{formula_max_abs_error:.9g}"
    )
    print(
        "self_loop_double_count_difference="
        f"{double_count_difference:.9g}"
    )
    print(
        "a1_sparse_graph_ops_estimate="
        f"{operation_estimate['a1_sparse_total']}"
    )
    print(
        "g1_sparse_graph_ops_estimate="
        f"{operation_estimate['g1_sparse_total']}"
    )
    print(
        "g1_vs_a1_sparse_ratio="
        f"{operation_estimate['g1_vs_a1_sparse_ratio']:.6f}"
    )
    print("training_started=False")
    print("validation_inference_performed=False")
    print("test_inference_performed=False")
    print(f"report={report_path}")

    if status != "EQUIVALENCE_AUDIT_PASSED":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"G0: FAIL — {failed}")

    print("G0 RESULT: GRAPHCONV IS MEANINGFULLY DIFFERENT")
    print("next_authorized_stage=G1 source generation and diff audit only")


if __name__ == "__main__":
    main()
