#!/usr/bin/env python3
"""
G1.1 real-data dry preflight for the bounded Chrono-G1 GraphConv diagnostic.

This stage performs one real training-batch forward/backward check only.
It performs no persistent training, writes no checkpoint, and performs no
validation or test inference.

It verifies:
- G1.0 passed;
- A1 and G1 source hashes match the audit;
- exact A1 split reuse;
- ordinary A1 shuffled training remains selected;
- GraphConv uses binary non-self adjacency with 48 directed entries;
- G1 has separate root and neighbor transforms;
- same-seed initialization/output determinism;
- real-batch graph/node losses and all gradients are finite;
- parameter count is 1,138;
- the planned G1 model directory is empty;
- all frozen A1 training arguments are captured for G1.2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_A1_SOURCE_SHA256 = (
    "2a70d2c6cac90f54504fa120126e643d9ce3a5d312436d6e59004e273d6d593b"
)
EXPECTED_A1_CHECKPOINT_SHA256 = (
    "c29cdd2303669a61454ed118ef0be4b8809a635b12c0c42806f8b6992340e0ef"
)
EXPECTED_G1_PARAMETER_COUNT = 1138

FORBIDDEN_TOKENS = (
    "chrono_b1_scenario_sampler",
    "scenario_balanced",
    "scenario-balanced",
    "weightedrandomsampler",
    "mean_plus_max",
    "mean-plus-max",
    "graph_max =",
)

EXPECTED_CLI_OPTIONS = (
    "--data",
    "--split-mode",
    "--epochs",
    "--patience",
    "--min-delta",
    "--batch-size",
    "--lr",
    "--weight-decay",
    "--temporal-dim",
    "--gcn-hidden",
    "--gcn-out",
    "--node-loss-weight",
    "--graph-threshold",
    "--node-threshold",
    "--seed",
)

MODEL_OUTPUT_OPTION_CANDIDATES = (
    "--model-dir",
    "--output-dir",
    "--out-dir",
    "--save-dir",
    "--model_dir",
    "--output_dir",
)

FROZEN_ARGUMENT_KEYS = (
    "split_mode",
    "epochs",
    "patience",
    "min_delta",
    "batch_size",
    "lr",
    "weight_decay",
    "temporal_dim",
    "gcn_hidden",
    "gcn_out",
    "node_loss_weight",
    "graph_threshold",
    "node_threshold",
    "seed",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_values(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(
                float(a),
                float(b),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False
    return a == b


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def positive_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives <= 0:
        raise ValueError("Cannot compute positive weight with zero positives.")
    return float(negatives / positives)


def manual_graphconv(
    layer: torch.nn.Module,
    h: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    neighbor_sum = torch.einsum(
        "ij,bjf->bif",
        adjacency,
        h,
    )
    return (
        layer.lin_neighbor(neighbor_sum)
        + layer.lin_root(h)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--a1-source", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--g1-source-audit", required=True, type=Path)
    parser.add_argument("--a1-model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--g1-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-batch-size", type=int, default=8)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    a1_source = args.a1_source.resolve()
    g1_source = args.g1_source.resolve()
    source_audit_path = args.g1_source_audit.resolve()
    a1_model_dir = args.a1_model_dir.resolve()
    data_dir = args.data_dir.resolve()
    g1_model_dir = args.g1_model_dir.resolve()
    output_dir = args.output_dir.resolve()

    a1_checkpoint = a1_model_dir / "best_model.pt"
    a1_splits = a1_model_dir / "splits.npz"

    required_files = {
        "a1_source": a1_source,
        "g1_source": g1_source,
        "g1_source_audit": source_audit_path,
        "a1_checkpoint": a1_checkpoint,
        "a1_splits": a1_splits,
        "x": data_dir / "x.npy",
        "y_graph": data_dir / "y_graph.npy",
        "y_node": data_dir / "y_node.npy",
        "edge_index": data_dir / "edge_index.npy",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing prerequisite {label}: {path}")

    if not repo_root.is_dir():
        raise SystemExit(f"STOP: repository root missing: {repo_root}")
    if not data_dir.is_dir():
        raise SystemExit(f"STOP: dataset directory missing: {data_dir}")
    if args.dry_batch_size <= 0:
        raise SystemExit("STOP: dry batch size must be positive.")
    if g1_model_dir.exists() and any(g1_model_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1 model directory already non-empty: {g1_model_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"STOP: G1.1 output directory already non-empty: {output_dir}"
        )

    source_audit = load_json(source_audit_path)
    audit_checks = source_audit.get("checks", {})
    prerequisite_checks = {
        "source_audit_stage_is_g1_0": (
            source_audit.get("stage") == "G1.0"
        ),
        "source_audit_status_passed": (
            source_audit.get("status") == "SOURCE_DIFF_AUDIT_PASSED"
        ),
        "source_audit_all_checks_true": (
            bool(audit_checks) and all(bool(v) for v in audit_checks.values())
        ),
        "source_audit_training_not_started": (
            source_audit.get("scope", {})
            .get("persistent_training_started") is False
        ),
        "source_audit_checkpoint_not_written": (
            source_audit.get("scope", {}).get("checkpoint_written") is False
        ),
        "source_audit_validation_not_inferred": (
            source_audit.get("scope", {})
            .get("validation_inference_performed") is False
        ),
        "source_audit_test_not_inferred": (
            source_audit.get("scope", {})
            .get("test_inference_performed") is False
        ),
        "a1_source_hash_frozen": (
            sha256(a1_source) == EXPECTED_A1_SOURCE_SHA256
        ),
        "a1_checkpoint_hash_frozen": (
            sha256(a1_checkpoint) == EXPECTED_A1_CHECKPOINT_SHA256
        ),
        "a1_source_hash_matches_audit": (
            sha256(a1_source)
            == source_audit["sources"]["a1_sha256"]
        ),
        "g1_source_hash_matches_audit": (
            sha256(g1_source)
            == source_audit["sources"]["g1_sha256"]
        ),
    }
    if not all(prerequisite_checks.values()):
        failed = [
            name for name, passed in prerequisite_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1.1 prerequisite checks failed: {failed}")

    source_text = g1_source.read_text(encoding="utf-8")
    lowered_source = source_text.lower()
    forbidden_hits = [
        token for token in FORBIDDEN_TOKENS
        if token in lowered_source
    ]

    source_structure_checks = {
        "no_b1_c1_sampler_or_readout_tokens": not forbidden_hits,
        "graphconv_class_present": "class GraphConvLayer" in source_text,
        "separate_neighbor_transform_present": (
            "self.lin_neighbor = nn.Linear" in source_text
        ),
        "separate_root_transform_present": (
            "self.lin_root = nn.Linear" in source_text
        ),
        "binary_adjacency_assignment_present": (
            "adjacency[target_i, source_i] = 1.0" in source_text
        ),
        "self_loop_exclusion_present": (
            "if source_i == target_i:" in source_text
            and "continue" in source_text
        ),
        "mean_graph_readout_retained": (
            "graph_embedding = h.mean(dim=1)" in source_text
        ),
        "meanmax_readout_absent": (
            "graph_max =" not in source_text
            and "torch.cat([graph_mean, graph_max]" not in source_text
        ),
        "g1_model_identifier_present": (
            "Conv1D-GraphConv-TemporalGNN" in source_text
        ),
    }
    if not all(source_structure_checks.values()):
        failed = [
            name for name, passed in source_structure_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1 source structure failed: {failed}")

    help_result = subprocess.run(
        [sys.executable, str(g1_source), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if help_result.returncode != 0:
        raise SystemExit(
            "STOP: G1 --help failed:\n"
            + help_result.stdout
            + "\n"
            + help_result.stderr
        )

    help_text = help_result.stdout + "\n" + help_result.stderr
    cli_option_checks = {
        option: option in help_text for option in EXPECTED_CLI_OPTIONS
    }
    if not all(cli_option_checks.values()):
        missing = [
            option for option, present in cli_option_checks.items()
            if not present
        ]
        raise SystemExit(f"STOP: G1 CLI options missing: {missing}")

    detected_output_options = [
        option
        for option in MODEL_OUTPUT_OPTION_CANDIDATES
        if option in help_text
    ]
    if len(detected_output_options) != 1:
        raise SystemExit(
            "STOP: expected exactly one model-output option; "
            f"found={detected_output_options}"
        )
    model_output_option = detected_output_options[0]

    checkpoint = torch.load(
        a1_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise SystemExit("STOP: A1 checkpoint has no saved args dictionary.")

    frozen_argument_checks: dict[str, bool] = {}
    frozen_arguments: dict[str, Any] = {}
    for key in FROZEN_ARGUMENT_KEYS:
        if key not in saved_args:
            frozen_argument_checks[key] = False
        else:
            frozen_arguments[key] = saved_args[key]
            frozen_argument_checks[key] = True

    if not all(frozen_argument_checks.values()):
        missing = [
            key for key, present in frozen_argument_checks.items()
            if not present
        ]
        raise SystemExit(f"STOP: A1 frozen arguments missing: {missing}")

    with np.load(a1_splits) as split_data:
        train_idx = np.asarray(split_data["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split_data["val_idx"], dtype=np.int64)
        test_idx = np.asarray(split_data["test_idx"], dtype=np.int64)

    x = np.load(data_dir / "x.npy", mmap_mode="r")
    y_graph = np.load(data_dir / "y_graph.npy", mmap_mode="r")
    y_node = np.load(data_dir / "y_node.npy", mmap_mode="r")
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)

    split_checks = {
        "train_unique": len(np.unique(train_idx)) == len(train_idx),
        "val_unique": len(np.unique(val_idx)) == len(val_idx),
        "test_unique": len(np.unique(test_idx)) == len(test_idx),
        "train_val_disjoint": len(np.intersect1d(train_idx, val_idx)) == 0,
        "train_test_disjoint": len(np.intersect1d(train_idx, test_idx)) == 0,
        "val_test_disjoint": len(np.intersect1d(val_idx, test_idx)) == 0,
        "all_indices_in_range": all(
            bool(np.all(idx >= 0) and np.all(idx < len(x)))
            for idx in (train_idx, val_idx, test_idx)
        ),
        "train_count_is_148185": len(train_idx) == 148185,
        "validation_count_is_42809": len(val_idx) == 42809,
        "test_count_is_42809": len(test_idx) == 42809,
    }
    if not all(split_checks.values()):
        failed = [name for name, passed in split_checks.items() if not passed]
        raise SystemExit(f"STOP: split checks failed: {failed}")

    dataset_checks = {
        "x_shape_is_233803_16_8_24": (
            tuple(x.shape) == (233803, 16, 8, 24)
        ),
        "y_graph_shape_is_233803": (
            tuple(y_graph.shape) == (233803,)
        ),
        "y_node_shape_is_233803_16": (
            tuple(y_node.shape) == (233803, 16)
        ),
        "edge_index_shape_is_2_64": (
            tuple(edge_index.shape) == (2, 64)
        ),
    }
    if not all(dataset_checks.values()):
        failed = [
            name for name, passed in dataset_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: dataset checks failed: {failed}")

    module = load_module(g1_source, "chrono_g1_real_data_preflight")
    required_symbols = (
        "TemporalGCN",
        "GraphConvLayer",
        "build_normalized_adjacency",
        "set_seed",
    )
    symbol_checks = {
        symbol: hasattr(module, symbol)
        for symbol in required_symbols
    }
    if not all(symbol_checks.values()):
        missing = [
            symbol for symbol, present in symbol_checks.items()
            if not present
        ]
        raise SystemExit(f"STOP: G1 source symbols missing: {missing}")

    adjacency = module.build_normalized_adjacency(
        edge_index,
        int(x.shape[1]),
    )
    adjacency_checks = {
        "adjacency_shape_is_16_16": (
            tuple(adjacency.shape) == (16, 16)
        ),
        "adjacency_is_binary": bool(
            torch.all((adjacency == 0) | (adjacency == 1)).item()
        ),
        "adjacency_has_zero_diagonal": bool(
            torch.all(torch.diag(adjacency) == 0).item()
        ),
        "adjacency_has_48_entries": (
            int(adjacency.sum().item()) == 48
        ),
        "adjacency_is_symmetric": bool(
            torch.equal(adjacency, adjacency.T)
        ),
    }
    if not all(adjacency_checks.values()):
        failed = [
            name for name, passed in adjacency_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1 adjacency checks failed: {failed}")

    train_graph_labels = np.asarray(
        y_graph[train_idx],
        dtype=np.int64,
    )
    train_node_labels = np.asarray(
        y_node[train_idx],
        dtype=np.int64,
    )
    graph_pos_weight = positive_weight(train_graph_labels)
    node_pos_weight = positive_weight(train_node_labels.reshape(-1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dry_count = min(args.dry_batch_size, len(train_idx))
    dry_indices = train_idx[:dry_count]

    batch_x = torch.from_numpy(
        np.array(x[dry_indices], dtype=np.float32, copy=True)
    ).to(device)
    batch_graph = torch.from_numpy(
        np.array(y_graph[dry_indices], dtype=np.float32, copy=True)
    ).to(device)
    batch_node = torch.from_numpy(
        np.array(y_node[dry_indices], dtype=np.float32, copy=True)
    ).to(device)
    adjacency_device = adjacency.to(device)

    seed = int(saved_args["seed"])
    module.set_seed(seed)
    model_a = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)

    module.set_seed(seed)
    model_b = module.TemporalGCN(
        input_features=int(x.shape[-1]),
        temporal_dim=int(saved_args["temporal_dim"]),
        gcn_hidden=int(saved_args["gcn_hidden"]),
        gcn_out=int(saved_args["gcn_out"]),
    ).to(device)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        graph_a, node_a = model_a(batch_x, adjacency_device)
        graph_b, node_b = model_b(batch_x, adjacency_device)

    deterministic_checks = {
        "same_seed_state_dict_equal": all(
            torch.equal(
                model_a.state_dict()[key].detach().cpu(),
                model_b.state_dict()[key].detach().cpu(),
            )
            for key in model_a.state_dict()
        ),
        "same_seed_graph_outputs_equal": torch.equal(graph_a, graph_b),
        "same_seed_node_outputs_equal": torch.equal(node_a, node_b),
    }

    with torch.no_grad():
        temporal_h = model_a.temporal(batch_x)
        manual_h1 = torch.relu(
            manual_graphconv(
                model_a.gcn1,
                temporal_h,
                adjacency_device,
            )
        )
        manual_h2 = torch.relu(
            manual_graphconv(
                model_a.gcn2,
                manual_h1,
                adjacency_device,
            )
        )
        expected_node = model_a.node_head(manual_h2).squeeze(-1)
        expected_graph = model_a.graph_head(
            manual_h2.mean(dim=1)
        ).squeeze(-1)
        actual_graph, actual_node = model_a(
            batch_x,
            adjacency_device,
        )

    manual_forward_checks = {
        "manual_graph_matches_forward": torch.equal(
            expected_graph,
            actual_graph,
        ),
        "manual_node_matches_forward": torch.equal(
            expected_node,
            actual_node,
        ),
    }

    model_a.train()
    model_a.zero_grad(set_to_none=True)
    graph_logits, node_logits = model_a(
        batch_x,
        adjacency_device,
    )

    graph_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            graph_pos_weight,
            device=device,
        )
    )
    node_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            node_pos_weight,
            device=device,
        )
    )

    graph_loss = graph_criterion(graph_logits, batch_graph)
    node_loss = node_criterion(node_logits, batch_node)
    total_loss = (
        graph_loss
        + float(saved_args["node_loss_weight"]) * node_loss
    )
    total_loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model_a.parameters()
        if parameter.requires_grad
    ]
    gradient_checks = {
        "all_trainable_parameters_have_gradients": all(
            parameter.grad is not None
            for parameter in trainable_parameters
        ),
        "all_gradients_finite": all(
            parameter.grad is not None
            and finite_tensor(parameter.grad)
            for parameter in trainable_parameters
        ),
        "at_least_one_nonzero_gradient": any(
            parameter.grad is not None
            and bool(torch.any(parameter.grad != 0).item())
            for parameter in trainable_parameters
        ),
    }

    state_keys = sorted(model_a.state_dict())
    parameter_count = sum(
        parameter.numel()
        for parameter in model_a.parameters()
    )
    runtime_checks = {
        "parameter_count_is_1138": (
            parameter_count == EXPECTED_G1_PARAMETER_COUNT
        ),
        "root_and_neighbor_state_keys_present": all(
            any(fragment in key for key in state_keys)
            for fragment in (
                "gcn1.lin_neighbor.weight",
                "gcn1.lin_root.weight",
                "gcn2.lin_neighbor.weight",
                "gcn2.lin_root.weight",
            )
        ),
        "graph_logits_shape_correct": (
            list(graph_logits.shape) == [dry_count]
        ),
        "node_logits_shape_correct": (
            list(node_logits.shape) == [dry_count, 16]
        ),
        "graph_logits_finite": finite_tensor(graph_logits),
        "node_logits_finite": finite_tensor(node_logits),
        "graph_loss_finite": finite_tensor(graph_loss),
        "node_loss_finite": finite_tensor(node_loss),
        "total_loss_finite": finite_tensor(total_loss),
        "model_directory_absent_or_empty": (
            not g1_model_dir.exists()
            or not any(g1_model_dir.iterdir())
        ),
    }

    all_checks = {
        **prerequisite_checks,
        **source_structure_checks,
        **{
            f"cli_{option}": present
            for option, present in cli_option_checks.items()
        },
        "exactly_one_model_output_option_detected": (
            len(detected_output_options) == 1
        ),
        **{
            f"frozen_arg_{key}": present
            for key, present in frozen_argument_checks.items()
        },
        **split_checks,
        **dataset_checks,
        **symbol_checks,
        **adjacency_checks,
        **deterministic_checks,
        **manual_forward_checks,
        **gradient_checks,
        **runtime_checks,
    }
    if not all(all_checks.values()):
        failed = [
            name for name, passed in all_checks.items()
            if not passed
        ]
        raise SystemExit(f"STOP: G1.1 preflight failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=False)

    training_plan = {
        "data": str(data_dir),
        "model_output_option": model_output_option,
        "model_output_path": str(g1_model_dir),
        **{
            key: saved_args[key]
            for key in FROZEN_ARGUMENT_KEYS
        },
        "sampler": "ordinary A1 shuffled training",
        "b1_sampler": False,
        "c1_readout": False,
        "graph_operator": (
            "separate-root-and-neighbor add-aggregation GraphConv"
        ),
        "source": str(g1_source),
        "source_sha256": sha256(g1_source),
        "a1_splits": str(a1_splits),
        "a1_splits_sha256": sha256(a1_splits),
    }
    plan_path = output_dir / "g1_frozen_training_arguments.json"
    plan_path.write_text(
        json.dumps(training_plan, indent=2) + "\n",
        encoding="utf-8",
    )

    help_path = output_dir / "g1_cli_help.txt"
    help_path.write_text(help_text, encoding="utf-8")

    report = {
        "stage": "G1.1",
        "status": "REAL_DATA_DRY_PREFLIGHT_PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "persistent_training_started": False,
        "checkpoint_written": False,
        "validation_inference_performed": False,
        "test_inference_performed": False,
        "test_threshold_selection_performed": False,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "paths": {
            "repo_root": str(repo_root),
            "a1_source": str(a1_source),
            "g1_source": str(g1_source),
            "g1_source_audit": str(source_audit_path),
            "a1_checkpoint": str(a1_checkpoint),
            "a1_splits": str(a1_splits),
            "data_dir": str(data_dir),
            "g1_model_dir": str(g1_model_dir),
        },
        "hashes": {
            "a1_source": sha256(a1_source),
            "g1_source": sha256(g1_source),
            "g1_source_audit": sha256(source_audit_path),
            "a1_checkpoint": sha256(a1_checkpoint),
            "a1_splits": sha256(a1_splits),
        },
        "checks": all_checks,
        "forbidden_token_hits": forbidden_hits,
        "cli": {
            "detected_model_output_option": model_output_option,
            "required_options": list(EXPECTED_CLI_OPTIONS),
            "detected_output_candidates": detected_output_options,
        },
        "dataset": {
            "x_shape": list(x.shape),
            "y_graph_shape": list(y_graph.shape),
            "y_node_shape": list(y_node.shape),
            "edge_index_shape": list(edge_index.shape),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "dry_batch_size": int(dry_count),
        },
        "model": {
            "name": "Conv1D-GraphConv-TemporalGNN",
            "parameter_count": parameter_count,
            "expected_parameter_count": EXPECTED_G1_PARAMETER_COUNT,
            "graph_readout": "mean",
            "graph_operator": (
                "separate root and neighbor transforms; add aggregation"
            ),
            "neighbor_adjacency_nonzero_entries": (
                int(adjacency.sum().item())
            ),
        },
        "loss": {
            "graph_pos_weight": graph_pos_weight,
            "node_pos_weight": node_pos_weight,
            "node_loss_weight": float(saved_args["node_loss_weight"]),
            "dry_graph_loss": float(graph_loss.detach().cpu()),
            "dry_node_loss": float(node_loss.detach().cpu()),
            "dry_total_loss": float(total_loss.detach().cpu()),
        },
        "frozen_training_plan": {
            "path": str(plan_path),
            "sha256": sha256(plan_path),
        },
    }

    report_path = output_dir / "g1_dry_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("G1.1 REAL-DATA DRY PREFLIGHT: PASS")
    print(f"device={device}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"g1_source_sha256={sha256(g1_source)}")
    print(f"model_output_cli_option={model_output_option}")
    print(f"b1_c1_forbidden_token_hits={len(forbidden_hits)}")
    print("ordinary_a1_sampler=True")
    print(f"split_integrity_pass={all(split_checks.values())}")
    print(f"train_rows={len(train_idx)}")
    print(f"validation_rows={len(val_idx)}")
    print(f"test_rows={len(test_idx)}")
    print(f"dry_batch_size={dry_count}")
    print(f"adjacency_zero_diagonal={adjacency_checks['adjacency_has_zero_diagonal']}")
    print(f"adjacency_nonzero_entries={int(adjacency.sum().item())}")
    print(f"parameter_count={parameter_count}")
    print(
        "same_seed_deterministic="
        f"{all(deterministic_checks.values())}"
    )
    print(
        "manual_graphconv_forward_matches="
        f"{all(manual_forward_checks.values())}"
    )
    print(
        "all_gradients_finite="
        f"{gradient_checks['all_gradients_finite']}"
    )
    print(f"dry_total_loss={float(total_loss.detach().cpu()):.6f}")
    print("g1_model_dir_empty=True")
    print("persistent_training_started=False")
    print("checkpoint_written=False")
    print("validation_inference_performed=False")
    print("test_inference_performed=False")
    print(f"training_plan={plan_path}")
    print(f"preflight_report={report_path}")
    print(f"cli_help={help_path}")


if __name__ == "__main__":
    main()
