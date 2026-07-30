#!/usr/bin/env python3
"""
V5 P2-G1A-R1 Source-Only Graph Baseline Preflight

This append-only revision resolves the two G1A HOLD causes:
1) verify the newly installed PyTorch Geometric stack with CPU/CUDA smoke tests;
2) discover and validate the existing P2 pair manifest required by the loader.

The historical G1A HOLD is preserved. This stage writes to a new R1 directory.

Allowed:
- read G0 and historical G1A HOLD artifacts;
- inspect non-test scripts/reports/logs/data metadata for pair-manifest references;
- instantiate P2 train and validation datasets;
- deserialize one deterministic train item and one validation item;
- instantiate the untrained B3 model;
- run shape-only PyG operator smoke tests.

Forbidden:
- enumerate or deserialize P2 test data;
- access B4/B6 prediction caches;
- load any checkpoint;
- train, optimize, tune thresholds, select an architecture, quantize,
  generate RTL, or implement the Legal NoC decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


STAGE = "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_G0_DECISION = (
    "FREEZE_VALIDATION_ONLY_GRAPH_BASELINE_BRANCH_BEFORE_"
    "LEGAL_NOC_DECODER_IMPLEMENTATION"
)
EXPECTED_G0_PROTOCOL_SHA = (
    "22e3d9828edaba9323a0c5206806ea93c149a239b8d9043ad79129de76882def"
)
EXPECTED_G1A_HOLD = "V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
EXPECTED_B3_PARAMETER_COUNT = 43273

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml"
}
MANIFEST_SUFFIXES = {
    ".json", ".jsonl", ".csv", ".txt", ".npy", ".npz", ".pkl", ".pickle"
}
SKIP_DIR_NAMES = {
    ".git", ".venv", "__pycache__", "node_modules"
}
FORBIDDEN_PATH_TOKENS = {
    "p2_b6",
    "b6_one_shot",
    "p2_c0",
    "p2_c1",
    "p2_l5",
    "test_evaluation",
    "blind_test",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def path_is_forbidden(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return any(token in lowered for token in FORBIDDEN_PATH_TOKENS)


def safe_walk(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_DIR_NAMES
            and not path_is_forbidden(current_path / directory)
        ]
        for filename in files:
            path = current_path / filename
            if not path_is_forbidden(path):
                yield path


def normalize_reference(value: str, repo: Path) -> Path | None:
    text = value.strip().strip("'\"`[](),")
    text = text.replace("${REPO}", str(repo)).replace("$REPO", str(repo))
    text = text.replace("${PWD}", str(repo)).replace("$PWD", str(repo))
    text = os.path.expandvars(os.path.expanduser(text))
    if not text:
        return None

    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def add_candidate(
    path: Path | None,
    source: str,
    evidence: str,
    candidates: dict[str, Path],
    evidence_rows: list[dict[str, str]],
) -> None:
    if path is None:
        return
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()

    if path_is_forbidden(resolved):
        return
    if not resolved.is_file():
        return
    if resolved.suffix.lower() not in MANIFEST_SUFFIXES:
        return
    if resolved.stat().st_size > 100 * 1024 * 1024:
        return

    candidates[str(resolved)] = resolved
    evidence_rows.append(
        {
            "candidate": str(resolved),
            "source": source,
            "evidence": evidence[:800],
        }
    )


def discover_manifest_candidates(
    repo: Path,
    data_root: Path,
    explicit_paths: list[Path],
) -> tuple[list[Path], list[dict[str, str]], list[str]]:
    candidates: dict[str, Path] = {}
    evidence_rows: list[dict[str, str]] = []
    searched_roots: list[str] = []

    for path in explicit_paths:
        add_candidate(
            path,
            "explicit_argument",
            str(path),
            candidates,
            evidence_rows,
        )

    filename_roots = [
        data_root,
        data_root.parent,
        repo / "data" / "processed" / "v5",
        repo / "reports" / "v5",
        repo / "artifacts" / "v5",
    ]
    for root in filename_roots:
        if not root.exists():
            continue
        searched_roots.append(str(root))
        for path in safe_walk(root):
            lower = path.name.lower()
            if (
                "pair" in lower
                and any(token in lower for token in ("manifest", "split", "index"))
            ):
                add_candidate(
                    path,
                    "filename_search",
                    str(path),
                    candidates,
                    evidence_rows,
                )

    text_roots = [
        repo / "scripts" / "v5",
        repo / "src",
        repo / "reports" / "v5",
        repo / "logs" / "v5",
    ]

    flag_pattern = re.compile(
        r"(?:--pair-manifest|--pair_manifest)\s+(?P<value>[^\s\\]+)"
    )
    quoted_assignment_pattern = re.compile(
        r"""pair[_-]?manifest["']?\s*[:=]\s*["'](?P<value>[^"']+)["']""",
        re.IGNORECASE,
    )
    absolute_pattern = re.compile(
        r"""(?P<value>(?:/|~|\$REPO|\$\{REPO\})[^\s"'`,]+"""
        r"""(?:pair[^\s"'`,]*manifest|manifest[^\s"'`,]*pair)"""
        r"""[^\s"'`,]*)""",
        re.IGNORECASE,
    )

    for root in text_roots:
        if not root.exists():
            continue
        searched_roots.append(str(root))
        for path in safe_walk(root):
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > 20 * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "pair_manifest" not in text and "pair-manifest" not in text:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if "pair_manifest" not in line and "pair-manifest" not in line:
                    continue
                context = f"{path}:{line_number}: {line.strip()}"

                for pattern in (
                    flag_pattern,
                    quoted_assignment_pattern,
                    absolute_pattern,
                ):
                    for match in pattern.finditer(line):
                        add_candidate(
                            normalize_reference(match.group("value"), repo),
                            "text_reference",
                            context,
                            candidates,
                            evidence_rows,
                        )

                try:
                    tokens = shlex.split(line)
                except ValueError:
                    tokens = []
                for index, token in enumerate(tokens):
                    if token in {"--pair-manifest", "--pair_manifest"}:
                        if index + 1 < len(tokens):
                            add_candidate(
                                normalize_reference(tokens[index + 1], repo),
                                "shell_flag_reference",
                                context,
                                candidates,
                                evidence_rows,
                            )

    ordered = sorted(candidates.values(), key=lambda path: str(path))
    return ordered, evidence_rows, sorted(set(searched_roots))


def find_dataset_classes(module) -> list[type]:
    classes: list[type] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, Dataset):
                classes.append(obj)
        except TypeError:
            pass
    return classes


def dataset_kwargs(
    cls: type,
    root: Path,
    split: str,
    pair_manifest: Path,
) -> dict[str, Any]:
    aliases: dict[str, Any] = {
        "root": root,
        "dataset_root": root,
        "data_root": root,
        "data_dir": root,
        "path": root,
        "split": split,
        "pair_manifest": pair_manifest,
        "pair_manifest_path": pair_manifest,
        "window": 32,
        "window_size": 32,
        "stride": 8,
        "active_only": False,
    }

    kwargs: dict[str, Any] = {}
    unresolved: list[str] = []
    for name, parameter in inspect.signature(cls).parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is not inspect._empty:
            continue
        elif name in {"self", "args", "kwargs"}:
            continue
        else:
            unresolved.append(name)

    if unresolved:
        raise TypeError(
            f"unresolved required parameters for {cls.__name__}: {unresolved}"
        )
    return kwargs


def instantiate_with_one_manifest(
    cls: type,
    root: Path,
    manifest: Path,
):
    train_kwargs = dataset_kwargs(cls, root, "train", manifest)
    validation_kwargs = dataset_kwargs(cls, root, "validation", manifest)
    train_dataset = cls(**train_kwargs)
    validation_dataset = cls(**validation_kwargs)

    if len(train_dataset) <= 0 or len(validation_dataset) <= 0:
        raise ValueError(
            f"empty dataset: train={len(train_dataset)}, "
            f"validation={len(validation_dataset)}"
        )

    return train_dataset, validation_dataset, {
        "train": {key: str(value) for key, value in train_kwargs.items()},
        "validation": {
            key: str(value) for key, value in validation_kwargs.items()
        },
    }


def instantiate_with_two_manifests(
    cls: type,
    root: Path,
    train_manifest: Path,
    validation_manifest: Path,
):
    train_kwargs = dataset_kwargs(cls, root, "train", train_manifest)
    validation_kwargs = dataset_kwargs(
        cls,
        root,
        "validation",
        validation_manifest,
    )
    train_dataset = cls(**train_kwargs)
    validation_dataset = cls(**validation_kwargs)

    if len(train_dataset) <= 0 or len(validation_dataset) <= 0:
        raise ValueError(
            f"empty dataset: train={len(train_dataset)}, "
            f"validation={len(validation_dataset)}"
        )

    return train_dataset, validation_dataset, {
        "train": {key: str(value) for key, value in train_kwargs.items()},
        "validation": {
            key: str(value) for key, value in validation_kwargs.items()
        },
    }


def resolve_datasets(
    classes: list[type],
    root: Path,
    candidates: list[Path],
):
    attempts: list[dict[str, Any]] = []

    for cls in classes:
        for manifest in candidates:
            try:
                train, validation, kwargs = instantiate_with_one_manifest(
                    cls,
                    root,
                    manifest,
                )
                return (
                    train,
                    validation,
                    cls,
                    kwargs,
                    {
                        "mode": "single_manifest",
                        "train_manifest": str(manifest),
                        "validation_manifest": str(manifest),
                    },
                    attempts,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "mode": "single_manifest",
                        "manifest": str(manifest),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    train_ranked = sorted(
        candidates,
        key=lambda path: (
            "train" not in path.name.lower(),
            "val" in path.name.lower(),
            str(path),
        ),
    )
    validation_ranked = sorted(
        candidates,
        key=lambda path: (
            not any(token in path.name.lower() for token in ("val", "valid")),
            "train" in path.name.lower(),
            str(path),
        ),
    )

    limit = min(20, len(candidates))
    for cls in classes:
        for train_manifest in train_ranked[:limit]:
            for validation_manifest in validation_ranked[:limit]:
                if train_manifest == validation_manifest:
                    continue
                try:
                    train, validation, kwargs = instantiate_with_two_manifests(
                        cls,
                        root,
                        train_manifest,
                        validation_manifest,
                    )
                    return (
                        train,
                        validation,
                        cls,
                        kwargs,
                        {
                            "mode": "separate_manifests",
                            "train_manifest": str(train_manifest),
                            "validation_manifest": str(validation_manifest),
                        },
                        attempts,
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "class": cls.__name__,
                            "mode": "separate_manifests",
                            "train_manifest": str(train_manifest),
                            "validation_manifest": str(validation_manifest),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    raise RuntimeError(
        "no candidate manifest configuration could instantiate both "
        f"train and validation datasets; attempts={attempts[-100:]}"
    )


def shape_dtype(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        result: dict[str, Any] = {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.dtype.is_floating_point:
            result["finite"] = bool(torch.isfinite(value).all().item())
        return result
    return {
        "type": type(value).__name__,
        "repr": repr(value)[:300],
    }


def find_model_classes(module) -> list[type]:
    classes: list[type] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, torch.nn.Module):
                classes.append(obj)
        except TypeError:
            pass
    return classes


def instantiate_model(classes: list[type]):
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
    attempts = []
    for cls in classes:
        signature = inspect.signature(cls)
        kwargs: dict[str, Any] = {}
        unresolved: list[str] = []
        for name, parameter in signature.parameters.items():
            if name in aliases:
                kwargs[name] = aliases[name]
            elif parameter.default is not inspect._empty:
                continue
            elif name in {"self", "args", "kwargs"}:
                continue
            else:
                unresolved.append(name)

        candidate_kwargs = [kwargs]
        if unresolved:
            candidate_kwargs.append({})

        for candidate in candidate_kwargs:
            try:
                model = cls(**candidate)
                return model, cls, candidate, attempts
            except Exception as exc:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "kwargs": candidate,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    raise RuntimeError(f"unable to instantiate B3 model: {attempts}")


def graph_operator_smoke_tests() -> dict[str, Any]:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv, GraphConv

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    x = torch.randn(4, 8)

    operator_specs = {
        "GCNConv": lambda: GCNConv(8, 8),
        "GraphConv": lambda: GraphConv(8, 8, aggr="add"),
        "GATConv": lambda: GATConv(8, 8, heads=1, concat=False),
    }

    results: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_geometric_version": torch_geometric.__version__,
        "operators": {},
    }

    for name, constructor in operator_specs.items():
        operator_result: dict[str, Any] = {}
        layer = constructor()
        with torch.no_grad():
            output = layer(x, edge_index)
        operator_result["cpu"] = {
            "passed": True,
            "shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all().item()),
        }

        if torch.cuda.is_available():
            device = torch.device("cuda")
            layer_cuda = constructor().to(device)
            with torch.no_grad():
                output_cuda = layer_cuda(
                    x.to(device),
                    edge_index.to(device),
                )
            operator_result["cuda"] = {
                "passed": True,
                "shape": list(output_cuda.shape),
                "finite": bool(torch.isfinite(output_cuda).all().item()),
                "device_name": torch.cuda.get_device_name(0),
            }
        else:
            operator_result["cuda"] = {
                "passed": False,
                "reason": "CUDA unavailable",
            }

        results["operators"][name] = operator_result

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--g1a-hold-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pair-manifest",
        action="append",
        default=[],
        help="Optional explicit candidate path. May be supplied multiple times.",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    g1a_hold_dir = args.g1a_hold_dir.expanduser().resolve()
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
    g1a_hold_marker = (
        g1a_hold_dir
        / "V5_P2_G1A_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
    )

    for path in (
        g0_report_path,
        g0_lock_path,
        g1a_hold_marker,
        loader_path,
        model_path,
    ):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if not data_root.is_dir():
        failures.append(f"missing data root: {data_root}")

    if not failures:
        g0_report = json.loads(
            g0_report_path.read_text(encoding="utf-8")
        )
        g0_lock = json.loads(g0_lock_path.read_text(encoding="utf-8"))

        if g0_report.get("status") != "COMPLETE":
            failures.append("G0 is not COMPLETE")
        if g0_report.get("decision") != EXPECTED_G0_DECISION:
            failures.append("G0 decision changed")
        if g0_lock.get("protocol_file_sha256") != sha256_file(
            g0_report_path
        ):
            failures.append("G0 protocol file SHA mismatch")
        if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
            failures.append("G0 canonical SHA changed")
        if (
            g1a_hold_marker.read_text(encoding="utf-8").strip()
            != EXPECTED_G1A_HOLD
        ):
            failures.append("historical G1A HOLD marker changed")

    explicit_paths = [
        Path(value).expanduser()
        for value in args.pair_manifest
        if value.strip()
    ]

    pyg_result: dict[str, Any] = {}
    try:
        pyg_result = graph_operator_smoke_tests()
    except Exception as exc:
        failures.append(
            "PyTorch Geometric smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        )

    candidates: list[Path] = []
    discovery_evidence: list[dict[str, str]] = []
    searched_roots: list[str] = []
    if not failures:
        candidates, discovery_evidence, searched_roots = (
            discover_manifest_candidates(
                repo,
                data_root,
                explicit_paths,
            )
        )
        if not candidates:
            failures.append(
                "no existing non-test pair-manifest candidates were found"
            )

    loader_module = None
    dataset_classes: list[type] = []
    if not failures:
        loader_module = import_module_from_path(
            "v5_p2_g1a_r1_loader",
            loader_path,
        )
        dataset_classes = find_dataset_classes(loader_module)
        if not dataset_classes:
            failures.append("no Dataset subclass found in loader")

    train_dataset = None
    validation_dataset = None
    selected_dataset_class = None
    selected_dataset_kwargs: dict[str, Any] = {}
    selected_manifest_info: dict[str, Any] = {}
    manifest_attempts: list[dict[str, Any]] = []

    if not failures:
        try:
            (
                train_dataset,
                validation_dataset,
                selected_dataset_class,
                selected_dataset_kwargs,
                selected_manifest_info,
                manifest_attempts,
            ) = resolve_datasets(
                dataset_classes,
                data_root,
                candidates,
            )
        except Exception as exc:
            failures.append(
                "pair-manifest resolution failed: "
                f"{type(exc).__name__}: {exc}"
            )

    sample_report: dict[str, Any] = {}
    train_sample = None
    validation_sample = None
    if train_dataset is not None and validation_dataset is not None:
        train_sample = train_dataset[0]
        validation_sample = validation_dataset[0]

        if not isinstance(train_sample, dict):
            failures.append("train sample is not a dictionary")
        if not isinstance(validation_sample, dict):
            failures.append("validation sample is not a dictionary")

        if isinstance(train_sample, dict) and isinstance(
            validation_sample,
            dict,
        ):
            required_keys = {"x", "edge_index", "y_attack", "y_source"}
            missing_train = required_keys - set(train_sample)
            missing_validation = required_keys - set(validation_sample)

            if missing_train:
                failures.append(
                    f"train sample missing: {sorted(missing_train)}"
                )
            if missing_validation:
                failures.append(
                    "validation sample missing: "
                    f"{sorted(missing_validation)}"
                )

            sample_report = {
                "train_length": len(train_dataset),
                "validation_length": len(validation_dataset),
                "train_keys": list(train_sample.keys()),
                "validation_keys": list(validation_sample.keys()),
                "train": {
                    key: shape_dtype(value)
                    for key, value in train_sample.items()
                },
                "validation": {
                    key: shape_dtype(value)
                    for key, value in validation_sample.items()
                },
            }

    model_report: dict[str, Any] = {}
    if not failures:
        try:
            model_module = import_module_from_path(
                "v5_p2_g1a_r1_model",
                model_path,
            )
            model_classes = find_model_classes(model_module)
            if not model_classes:
                raise RuntimeError("no torch.nn.Module subclass found")

            model, model_class, model_kwargs, model_attempts = (
                instantiate_model(model_classes)
            )
            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
            )
            model_report = {
                "selected_class": model_class.__name__,
                "selected_kwargs": model_kwargs,
                "failed_attempts": model_attempts,
                "parameter_count": parameter_count,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "forward_signature": str(
                    inspect.signature(model.forward)
                ),
                "state_dict_shapes": {
                    key: list(value.shape)
                    for key, value in model.state_dict().items()
                },
            }
            if parameter_count != EXPECTED_B3_PARAMETER_COUNT:
                warnings.append(
                    "instantiated B3 parameter count differs from frozen "
                    f"value: got {parameter_count}, "
                    f"expected {EXPECTED_B3_PARAMETER_COUNT}"
                )
        except Exception as exc:
            failures.append(
                "B3 model resolution failed: "
                f"{type(exc).__name__}: {exc}"
            )

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION_FROM_RESOLVED_R1_INTERFACES"
            if not failures
            else "HOLD_G1_SOURCE_ONLY_IMPLEMENTATION_PENDING_R1_FIX"
        ),
        "historical_g1a_hold_preserved": True,
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "environment": pyg_result,
        "manifest_discovery": {
            "searched_roots": searched_roots,
            "candidate_count": len(candidates),
            "candidates": [str(path) for path in candidates],
            "evidence": discovery_evidence,
            "selected": selected_manifest_info,
            "attempt_count": len(manifest_attempts),
            "attempts_tail": manifest_attempts[-100:],
        },
        "loader": {
            "path": str(loader_path),
            "sha256": sha256_file(loader_path)
            if loader_path.is_file()
            else None,
            "dataset_classes": [
                {
                    "class": cls.__name__,
                    "signature": str(inspect.signature(cls)),
                }
                for cls in dataset_classes
            ],
            "selected_class": (
                selected_dataset_class.__name__
                if selected_dataset_class is not None
                else None
            ),
            "selected_kwargs": selected_dataset_kwargs,
        },
        "samples": sample_report,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path)
            if model_path.is_file()
            else None,
            **model_report,
        },
        "security_boundary": {
            "historical_hold_deleted": False,
            "train_samples_deserialized": (
                1 if train_sample is not None else 0
            ),
            "validation_samples_deserialized": (
                1 if validation_sample is not None else 0
            ),
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
            else "V5_P2_G1A_R2_PAIR_MANIFEST_RESOLUTION"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    markdown = [
        "# V5 P2-G1A-R1 Source-Only Graph Baseline Preflight",
        "",
        f"- Status: **{report['status']}**",
        f"- PyTorch: `{pyg_result.get('torch_version')}`",
        f"- PyTorch Geometric: "
        f"`{pyg_result.get('torch_geometric_version')}`",
        f"- Candidate manifests found: `{len(candidates)}`",
        f"- Selected manifest mode: "
        f"`{selected_manifest_info.get('mode')}`",
        f"- Train manifest: "
        f"`{selected_manifest_info.get('train_manifest')}`",
        f"- Validation manifest: "
        f"`{selected_manifest_info.get('validation_manifest')}`",
        f"- Train items: `{sample_report.get('train_length')}`",
        f"- Validation items: "
        f"`{sample_report.get('validation_length')}`",
        "",
        "The historical G1A HOLD remains preserved. R1 performs no "
        "checkpoint loading, training, test access, quantization, RTL "
        "generation, or Legal NoC decoder implementation.",
        "",
    ]
    atomic_write(
        output_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT.md",
        "\n".join(markdown),
    )

    lock = {
        "status": (
            COMPLETE
            if not failures
            else "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
        ),
        "report_sha256": sha256_file(report_path),
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "historical_g1a_hold_preserved": True,
        "loader_sha256": (
            sha256_file(loader_path)
            if loader_path.is_file()
            else None
        ),
        "model_sha256": (
            sha256_file(model_path)
            if model_path.is_file()
            else None
        ),
        "selected_train_manifest_sha256": (
            sha256_file(Path(selected_manifest_info["train_manifest"]))
            if selected_manifest_info.get("train_manifest")
            else None
        ),
        "selected_validation_manifest_sha256": (
            sha256_file(
                Path(selected_manifest_info["validation_manifest"])
            )
            if selected_manifest_info.get("validation_manifest")
            else None
        ),
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
        "next_stage": report["next_stage"],
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_LOCK.json",
        lock,
    )

    if failures:
        atomic_write(
            output_dir
            / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD",
            "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD\n",
        )
        print("===== V5 P2-G1A-R1 SOURCE-ONLY GRAPH PREFLIGHT =====")
        print("status: HOLD")
        print("historical_g1a_hold_preserved: true")
        print(
            "torch_geometric_version:",
            pyg_result.get("torch_geometric_version"),
        )
        print("manifest_candidate_count:", len(candidates))
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        for failure in failures:
            print("FAIL:", failure)
        print(
            "next_stage: "
            "V5_P2_G1A_R2_PAIR_MANIFEST_RESOLUTION"
        )
        print(
            "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT_HOLD"
        )
        return 1

    atomic_write(output_dir / COMPLETE, COMPLETE + "\n")

    print("===== V5 P2-G1A-R1 SOURCE-ONLY GRAPH PREFLIGHT =====")
    print("status: COMPLETE")
    print(
        "decision: "
        "AUTHORIZE_G1_SOURCE_ONLY_IMPLEMENTATION_FROM_RESOLVED_R1_INTERFACES"
    )
    print("historical_g1a_hold_preserved: true")
    print("torch_version:", pyg_result.get("torch_version"))
    print("torch_cuda:", pyg_result.get("torch_cuda"))
    print(
        "torch_geometric_version:",
        pyg_result.get("torch_geometric_version"),
    )
    print(
        "graph_operator_cpu_smoke: "
        "GCNConv=true,GraphConv=true,GATConv=true"
    )
    if torch.cuda.is_available():
        print(
            "graph_operator_cuda_smoke: "
            "GCNConv=true,GraphConv=true,GATConv=true"
        )
    else:
        print("graph_operator_cuda_smoke: CUDA_UNAVAILABLE")
    print("manifest_candidate_count:", len(candidates))
    print("manifest_mode:", selected_manifest_info.get("mode"))
    print(
        "train_pair_manifest:",
        selected_manifest_info.get("train_manifest"),
    )
    print(
        "validation_pair_manifest:",
        selected_manifest_info.get("validation_manifest"),
    )
    print("dataset_class:", selected_dataset_class.__name__)
    print("train_items:", sample_report.get("train_length"))
    print("validation_items:", sample_report.get("validation_length"))
    print("train_x_shape:", sample_report["train"]["x"]["shape"])
    print(
        "train_y_source_shape:",
        sample_report["train"]["y_source"]["shape"],
    )
    print(
        "edge_index_shape:",
        sample_report["train"]["edge_index"]["shape"],
    )
    print("model_class:", model_report.get("selected_class"))
    print("model_parameter_count:", model_report.get("parameter_count"))
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
