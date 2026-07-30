#!/usr/bin/env python3
"""
V5 P2-G1A Source-Only Graph Baseline Preflight

Purpose:
Resolve the exact P2 loader, sample tensor interface, B3 model constructor,
temporal encoder structure, graph topology, and installed graph-operator
environment before any G1 training code is frozen.

Allowed:
- read G0 protocol artifacts;
- import the P2 train/validation loader;
- deserialize one deterministic train sample and one validation sample;
- import/instantiate the B3 model;
- run shape-only forward probes on those samples when possible;
- inspect torch/torch-geometric versions and operator signatures.

Forbidden:
- enumerate the test directory;
- deserialize test tensors;
- load any P2 checkpoint;
- access B4 or B6 caches;
- train, optimize, tune thresholds, select architectures, quantize, or
  generate RTL;
- implement the Legal NoC decoder.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


STAGE = "V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT"
COMPLETE = f"{STAGE}_COMPLETE"
EXPECTED_G0_DECISION = (
    "FREEZE_VALIDATION_ONLY_GRAPH_BASELINE_BRANCH_BEFORE_"
    "LEGAL_NOC_DECODER_IMPLEMENTATION"
)
EXPECTED_G0_PROTOCOL_SHA = (
    "22e3d9828edaba9323a0c5206806ea93c149a239b8d9043ad79129de76882def"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shape_dtype(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item())
            if value.dtype.is_floating_point
            else None,
        }
    return {
        "type": type(value).__name__,
        "repr": repr(value)[:300],
    }


def find_dataset_classes(module) -> list[type]:
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, Dataset):
                classes.append(obj)
        except TypeError:
            pass
    return classes


def construct_dataset(cls, root: Path, split: str):
    sig = inspect.signature(cls)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    aliases = {
        "root": root,
        "dataset_root": root,
        "data_root": root,
        "data_dir": root,
        "path": root,
        "split": split,
        "window": 32,
        "window_size": 32,
        "stride": 8,
        "active_only": False,
    }

    unresolved = []
    for name, param in params.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif param.default is not inspect._empty:
            continue
        elif name in {"self", "args", "kwargs"}:
            continue
        else:
            unresolved.append(name)

    if unresolved:
        raise TypeError(
            f"unresolved required parameters for {cls.__name__}: {unresolved}"
        )
    return cls(**kwargs), kwargs


def find_model_classes(module) -> list[type]:
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, torch.nn.Module):
                classes.append(obj)
        except TypeError:
            pass
    return classes


def candidate_model_kwargs(cls) -> list[dict[str, Any]]:
    sig = inspect.signature(cls)
    aliases = {
        "num_features": 58,
        "in_features": 58,
        "input_features": 58,
        "feature_dim": 58,
        "input_dim": 58,
        "num_nodes": 16,
        "n_nodes": 16,
        "window": 32,
        "window_size": 32,
        "count_classes": 4,
        "num_count_classes": 4,
        "num_attacker_count_classes": 4,
    }
    base: dict[str, Any] = {}
    unresolved = []
    for name, param in sig.parameters.items():
        if name in aliases:
            base[name] = aliases[name]
        elif param.default is not inspect._empty:
            continue
        elif name in {"self", "args", "kwargs"}:
            continue
        else:
            unresolved.append(name)

    candidates = [base]
    if unresolved:
        candidates.append({})
    return candidates


def instantiate_first_model(classes: list[type]):
    attempts = []
    for cls in classes:
        for kwargs in candidate_model_kwargs(cls):
            try:
                model = cls(**kwargs)
                return model, cls, kwargs, attempts
            except Exception as exc:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "kwargs": kwargs,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    raise RuntimeError(f"unable to instantiate any model class: {attempts}")


def try_forward(model, sample: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    x = sample["x"]
    edge_index = sample.get("edge_index")
    if x.ndim == 3:
        x_batched = x.unsqueeze(0)
    else:
        x_batched = x

    sig = inspect.signature(model.forward)
    attempts = []
    call_variants = []
    names = list(sig.parameters.keys())

    if "edge_index" in names:
        ei = edge_index
        if torch.is_tensor(ei) and ei.ndim == 2:
            call_variants.append(("x_edge_index", (x_batched, ei), {}))
        call_variants.append(
            ("keyword", (), {"x": x_batched, "edge_index": edge_index})
        )
    call_variants.extend(
        [
            ("x_only", (x_batched,), {}),
            ("keyword_x", (), {"x": x_batched}),
        ]
    )

    for name, args, kwargs in call_variants:
        try:
            with torch.no_grad():
                output = model(*args, **kwargs)
            return {
                "success": True,
                "variant": name,
                "output": describe_output(output),
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append(
                {
                    "variant": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "success": False,
        "forward_signature": str(sig),
        "attempts": attempts,
    }


def describe_output(output: Any) -> Any:
    if torch.is_tensor(output):
        return shape_dtype(output)
    if isinstance(output, dict):
        return {str(k): shape_dtype(v) for k, v in output.items()}
    if isinstance(output, (tuple, list)):
        return [shape_dtype(v) for v in output]
    return {"type": type(output).__name__, "repr": repr(output)[:500]}


def module_tree(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        row = {
            "name": name,
            "class": module.__class__.__name__,
        }
        if isinstance(module, torch.nn.Conv1d):
            row.update(
                {
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "dilation": list(module.dilation),
                    "groups": module.groups,
                    "bias": module.bias is not None,
                }
            )
        elif isinstance(module, torch.nn.Linear):
            row.update(
                {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                }
            )
        rows.append(row)
    return rows


def parse_source_classes(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append(
                {
                    "class": node.name,
                    "bases": [
                        ast.unparse(base)
                        if hasattr(ast, "unparse")
                        else type(base).__name__
                        for base in node.bases
                    ],
                    "methods": [
                        child.name
                        for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    loader_path = args.loader.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    g0_report_path = (
        g0_dir / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL.json"
    )
    g0_lock_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    g0_complete_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_COMPLETE"
    )

    for path in (
        g0_report_path,
        g0_lock_path,
        g0_complete_path,
        loader_path,
        model_path,
    ):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if not data_root.is_dir():
        failures.append(f"missing P2 data root: {data_root}")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        return 1

    g0_report = json.loads(g0_report_path.read_text(encoding="utf-8"))
    g0_lock = json.loads(g0_lock_path.read_text(encoding="utf-8"))

    if g0_report.get("status") != "COMPLETE":
        failures.append("G0 report is not COMPLETE")
    if g0_report.get("decision") != EXPECTED_G0_DECISION:
        failures.append("G0 decision changed")
    if g0_lock.get("protocol_file_sha256") != sha256_file(g0_report_path):
        failures.append("G0 protocol file SHA mismatch")
    if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
        failures.append("G0 canonical protocol SHA changed")
    if g0_lock.get("p2_test_tensor_contents_accessed") is not False:
        failures.append("G0 unexpectedly accessed P2 test tensors")

    if failures:
        report = {
            "stage": STAGE,
            "status": "HOLD",
            "failures": failures,
            "warnings": warnings,
        }
        write_json(output_dir / f"{STAGE}.json", report)
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print(f"{STAGE}_HOLD")
        for failure in failures:
            print("FAIL:", failure)
        return 1

    environment = {
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_geometric_available": False,
        "torch_geometric_version": None,
        "operators": {},
    }

    try:
        import torch_geometric
        from torch_geometric.nn import GATConv, GCNConv, GraphConv

        environment["torch_geometric_available"] = True
        environment["torch_geometric_version"] = torch_geometric.__version__
        for cls in (GCNConv, GraphConv, GATConv):
            environment["operators"][cls.__name__] = {
                "signature": str(inspect.signature(cls)),
                "forward_signature": str(inspect.signature(cls.forward)),
                "module": cls.__module__,
            }
    except Exception as exc:
        failures.append(
            "torch_geometric graph operators unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    loader_module = import_module_from_path("v5_p2_g1a_loader", loader_path)
    dataset_classes = find_dataset_classes(loader_module)
    if not dataset_classes:
        failures.append("no torch Dataset subclass found in P2 loader")

    dataset_attempts = []
    train_dataset = None
    validation_dataset = None
    selected_dataset_class = None
    selected_dataset_kwargs = {}

    for cls in dataset_classes:
        try:
            train_dataset, train_kwargs = construct_dataset(
                cls, data_root, "train"
            )
            validation_dataset, val_kwargs = construct_dataset(
                cls, data_root, "validation"
            )
            selected_dataset_class = cls
            selected_dataset_kwargs = {
                "train": {k: str(v) for k, v in train_kwargs.items()},
                "validation": {k: str(v) for k, v in val_kwargs.items()},
            }
            break
        except Exception as exc:
            dataset_attempts.append(
                {
                    "class": cls.__name__,
                    "signature": str(inspect.signature(cls)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if train_dataset is None or validation_dataset is None:
        failures.append(
            "unable to construct train/validation datasets: "
            f"{dataset_attempts}"
        )

    train_sample = None
    validation_sample = None
    sample_report = {}
    if train_dataset is not None and validation_dataset is not None:
        if len(train_dataset) == 0 or len(validation_dataset) == 0:
            failures.append("train or validation dataset is empty")
        else:
            train_sample = train_dataset[0]
            validation_sample = validation_dataset[0]
            if not isinstance(train_sample, dict):
                failures.append("train sample is not a dict")
            if not isinstance(validation_sample, dict):
                failures.append("validation sample is not a dict")
            if isinstance(train_sample, dict) and isinstance(
                validation_sample, dict
            ):
                required = {"x", "edge_index", "y_attack", "y_source"}
                missing_train = required - set(train_sample)
                missing_val = required - set(validation_sample)
                if missing_train:
                    failures.append(
                        f"train sample missing keys: {sorted(missing_train)}"
                    )
                if missing_val:
                    failures.append(
                        "validation sample missing keys: "
                        f"{sorted(missing_val)}"
                    )
                sample_report = {
                    "train_length": len(train_dataset),
                    "validation_length": len(validation_dataset),
                    "train_keys": list(train_sample.keys()),
                    "validation_keys": list(validation_sample.keys()),
                    "train": {
                        k: shape_dtype(v) for k, v in train_sample.items()
                    },
                    "validation": {
                        k: shape_dtype(v)
                        for k, v in validation_sample.items()
                    },
                }

    model_module = import_module_from_path("v5_p2_g1a_model", model_path)
    model_classes = find_model_classes(model_module)
    if not model_classes:
        failures.append("no torch.nn.Module subclass found in B3 model source")

    model_info: dict[str, Any] = {
        "source_classes": parse_source_classes(model_path),
        "candidate_classes": [
            {
                "class": cls.__name__,
                "signature": str(inspect.signature(cls)),
                "forward_signature": str(inspect.signature(cls.forward)),
            }
            for cls in model_classes
        ],
    }

    selected_model = None
    if model_classes:
        try:
            selected_model, selected_cls, kwargs, attempts = (
                instantiate_first_model(model_classes)
            )
            model_info.update(
                {
                    "selected_class": selected_cls.__name__,
                    "selected_kwargs": kwargs,
                    "failed_instantiation_attempts": attempts,
                    "parameter_count": sum(
                        p.numel() for p in selected_model.parameters()
                    ),
                    "trainable_parameter_count": sum(
                        p.numel()
                        for p in selected_model.parameters()
                        if p.requires_grad
                    ),
                    "module_tree": module_tree(selected_model),
                    "state_dict": {
                        k: {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                        }
                        for k, v in selected_model.state_dict().items()
                    },
                }
            )
            if train_sample is not None and isinstance(train_sample, dict):
                model_info["forward_probe"] = try_forward(
                    selected_model, train_sample
                )
        except Exception as exc:
            failures.append(
                "unable to instantiate/probe B3 model: "
                f"{type(exc).__name__}: {exc}"
            )
            model_info["instantiation_traceback"] = traceback.format_exc()

    topology_checks: dict[str, Any] = {}
    if isinstance(train_sample, dict) and torch.is_tensor(
        train_sample.get("edge_index")
    ):
        edge_index = train_sample["edge_index"].detach().cpu()
        topology_checks = {
            "shape": list(edge_index.shape),
            "dtype": str(edge_index.dtype),
            "min": int(edge_index.min().item()),
            "max": int(edge_index.max().item()),
            "directed_edge_count": int(edge_index.shape[1])
            if edge_index.ndim == 2
            else None,
            "unique_directed_edges": int(
                torch.unique(edge_index.t(), dim=0).shape[0]
            )
            if edge_index.ndim == 2
            else None,
            "self_loop_count": int(
                (edge_index[0] == edge_index[1]).sum().item()
            )
            if edge_index.ndim == 2
            else None,
        }
        if topology_checks["shape"] != [2, 48]:
            failures.append(
                f"unexpected edge_index shape: {topology_checks['shape']}"
            )
        if topology_checks["min"] != 0 or topology_checks["max"] != 15:
            failures.append("edge_index node range is not 0..15")
        if topology_checks["self_loop_count"] != 0:
            warnings.append("stored topology contains self-loops")

    resolved = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION_FROM_RESOLVED_INTERFACES"
            if not failures
            else "HOLD_G1_SOURCE_ONLY_IMPLEMENTATION_PENDING_INTERFACE_FIX"
        ),
        "repo": str(repo),
        "data_root": str(data_root),
        "loader": {
            "path": str(loader_path),
            "sha256": sha256_file(loader_path),
            "source_classes": parse_source_classes(loader_path),
            "dataset_candidates": [
                {
                    "class": cls.__name__,
                    "signature": str(inspect.signature(cls)),
                }
                for cls in dataset_classes
            ],
            "selected_class": selected_dataset_class.__name__
            if selected_dataset_class
            else None,
            "selected_kwargs": selected_dataset_kwargs,
            "failed_attempts": dataset_attempts,
        },
        "samples": sample_report,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            **model_info,
        },
        "topology": topology_checks,
        "environment": environment,
        "implementation_requirements": {
            "source_target_key": "y_source",
            "graph_target_key": "y_attack",
            "required_input_key": "x",
            "required_topology_key": "edge_index",
            "graph_layer_count": 2,
            "mandatory_models": [
                "CONV1D_ONLY",
                "CONV1D_PLUS_GCNCONV",
                "CONV1D_PLUS_GRAPHCONV",
            ],
            "screening_model": "CONV1D_PLUS_GAT",
            "initial_seeds": [107, 117, 127],
            "test_access": False,
        },
        "security_boundary": {
            "train_samples_deserialized": 1,
            "validation_samples_deserialized": 1,
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "training_performed": False,
            "optimization_steps": 0,
            "threshold_tuning_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
            "legal_decoder_implemented": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
            if not failures
            else "V5_P2_G1A_R1_INTERFACE_RESOLUTION"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, resolved)

    markdown = [
        "# V5 P2-G1A Source-Only Graph Baseline Preflight",
        "",
        f"- Status: **{resolved['status']}**",
        f"- Loader: `{loader_path}`",
        f"- Model: `{model_path}`",
        f"- Train items: `{sample_report.get('train_length')}`",
        f"- Validation items: `{sample_report.get('validation_length')}`",
        f"- Dataset class: `{resolved['loader']['selected_class']}`",
        f"- Model class: `{model_info.get('selected_class')}`",
        f"- PyTorch: `{environment['torch_version']}`",
        f"- PyTorch Geometric: `{environment['torch_geometric_version']}`",
        "",
        "This stage only resolves interfaces. It does not train, load a "
        "checkpoint, access the P2 test, tune thresholds, quantize, generate "
        "RTL, or implement the Legal NoC decoder.",
        "",
    ]
    atomic_write(
        output_dir / "V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT.md",
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE if not failures else f"{STAGE}_HOLD",
        "report_sha256": sha256_file(report_path),
        "loader_sha256": sha256_file(loader_path),
        "model_sha256": sha256_file(model_path),
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
        "checkpoint_loaded": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "training_performed": False,
        "optimization_steps": 0,
        "architecture_selected": False,
        "quantization_performed": False,
        "rtl_generated": False,
        "legal_decoder_implemented": False,
        "next_stage": resolved["next_stage"],
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir
        / "V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_LOCK.json",
        lock,
    )

    if failures:
        atomic_write(output_dir / f"{STAGE}_HOLD", f"{STAGE}_HOLD\n")
        print("===== V5 P2-G1A SOURCE-ONLY GRAPH PREFLIGHT =====")
        print("status: HOLD")
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        for failure in failures:
            print("FAIL:", failure)
        print(f"{STAGE}_HOLD")
        return 1

    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G1A SOURCE-ONLY GRAPH PREFLIGHT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION_FROM_RESOLVED_INTERFACES"
    )
    print("dataset_class:", resolved["loader"]["selected_class"])
    print("train_items:", sample_report.get("train_length"))
    print("validation_items:", sample_report.get("validation_length"))
    print("train_x_shape:", sample_report["train"]["x"]["shape"])
    print("train_y_source_shape:", sample_report["train"]["y_source"]["shape"])
    print("edge_index_shape:", topology_checks.get("shape"))
    print("model_class:", model_info.get("selected_class"))
    print("model_parameter_count:", model_info.get("parameter_count"))
    print(
        "model_forward_probe_success:",
        model_info.get("forward_probe", {}).get("success"),
    )
    print("torch_version:", environment["torch_version"])
    print(
        "torch_geometric_version:",
        environment["torch_geometric_version"],
    )
    print(
        "available_graph_operators:",
        ",".join(sorted(environment["operators"].keys())),
    )
    print("train_samples_deserialized: 1")
    print("validation_samples_deserialized: 1")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print("checkpoint_loaded: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("training_performed: false")
    print("optimization_steps: 0")
    print("architecture_selected: false")
    print("failure_count: 0")
    print("warning_count:", len(warnings))
    print(
        "next_stage: "
        "V5_P2_G1_SOURCE_ONLY_GRAPH_BASELINE_IMPLEMENTATION_AND_TRAINING"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
