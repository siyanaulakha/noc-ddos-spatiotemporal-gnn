"""
V5 P2-G1 source-only graph implementation smoke check.

This is the consolidated smoke stage for Conv1D-only, GCNConv, GraphConv,
and GATConv source-localization baselines. It intentionally depends only on
the frozen G0 protocol, the G1A-R1 manifest resolution, the canonical 4x4
topology contract, and the P2 train/validation data. Historical failed smoke
revisions remain untouched but are not prerequisites.

The script performs one optimizer step per architecture. It never opens the
P2 test split, loads a checkpoint, accesses B4/B6 caches, tunes thresholds,
selects an architecture, quantizes, generates RTL, or implements the Legal
NoC decoder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torch_geometric.nn import GATConv, GCNConv, GraphConv
    PYG_IMPORT_ERROR: str | None = None
except Exception as exc:  # Converted to a normal HOLD inside run_smoke.
    GATConv = None
    GCNConv = None
    GraphConv = None
    PYG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


STAGE = "V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK"
COMPLETE = f"{STAGE}_COMPLETE"

EXPECTED_G0_PROTOCOL_SHA = (
    "22e3d9828edaba9323a0c5206806ea93c149a239b8d9043ad79129de76882def"
)
EXPECTED_EDGE_ARTIFACT_SHA = (
    "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff"
)
EXPECTED_B3_PARAMETER_COUNT = 43273


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_dataset_class(module, class_name: str | None):
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, Dataset):
                candidates.append(obj)
        except TypeError:
            pass

    if class_name:
        for cls in candidates:
            if cls.__name__ == class_name:
                return cls
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "unable to select dataset class: "
        f"requested={class_name}, "
        f"candidates={[cls.__name__ for cls in candidates]}"
    )


def build_dataset_kwargs(
    cls,
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
            f"unresolved dataset constructor parameters: {unresolved}"
        )
    return kwargs



def source_rows(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 1 and tensor.shape[0] == 16:
        return tensor.reshape(1, 16)
    if tensor.ndim >= 2 and tensor.shape[-1] == 16:
        return tensor.reshape(-1, 16)
    raise RuntimeError(
        f"source target does not end in 16 routers: shape={tuple(tensor.shape)}"
    )


def source_row_profile(value: Any) -> dict[str, int]:
    rows = source_rows(value).float()
    positive = (rows.sum(dim=1) > 0)
    control = ~positive
    return {
        "rows": int(rows.shape[0]),
        "attack_rows": int(positive.sum().item()),
        "control_rows": int(control.sum().item()),
    }


def select_balanced_smoke_indices(
    dataset: Dataset,
    requested_items: int,
    max_scan: int = 1024,
) -> tuple[list[int], dict[str, Any]]:
    if requested_items < 2:
        raise ValueError("requested_items must be at least 2")

    attack_only: list[int] = []
    control_only: list[int] = []
    mixed: list[int] = []
    scanned = 0

    scan_limit = min(len(dataset), max_scan)
    for index in range(scan_limit):
        item = dataset[index]
        if not isinstance(item, dict) or "y_source" not in item:
            raise RuntimeError(
                f"dataset item {index} has no dictionary y_source target"
            )

        profile = source_row_profile(item["y_source"])
        has_attack = profile["attack_rows"] > 0
        has_control = profile["control_rows"] > 0

        if has_attack and has_control:
            mixed.append(index)
        elif has_attack:
            attack_only.append(index)
        elif has_control:
            control_only.append(index)
        else:
            raise RuntimeError(
                f"dataset item {index} contains no source-label rows"
            )

        scanned += 1
        enough_items = (
            len(attack_only) + len(control_only) + len(mixed)
            >= requested_items
        )
        has_both_classes = bool(mixed) or (
            bool(attack_only) and bool(control_only)
        )
        if enough_items and has_both_classes:
            break

    has_both_classes = bool(mixed) or (
        bool(attack_only) and bool(control_only)
    )
    if not has_both_classes:
        raise RuntimeError(
            "could not find both attack and control source rows after "
            f"scanning {scanned} dataset items"
        )

    selected: list[int] = []

    # Put class coverage at the beginning of the subset.
    if mixed:
        selected.append(mixed.pop(0))
    else:
        selected.extend([
            control_only.pop(0),
            attack_only.pop(0),
        ])

    # Fill deterministically while alternating available categories.
    while len(selected) < requested_items:
        added = False
        for pool in (attack_only, control_only, mixed):
            if pool and len(selected) < requested_items:
                selected.append(pool.pop(0))
                added = True
        if not added:
            break

    if len(selected) != requested_items:
        raise RuntimeError(
            "found both classes but could not build the requested compact "
            f"smoke set: selected={len(selected)}, "
            f"requested={requested_items}, scanned={scanned}"
        )

    selected_profile = {
        "attack_items": 0,
        "control_items": 0,
        "mixed_items": 0,
    }
    for index in selected:
        profile = source_row_profile(dataset[index]["y_source"])
        has_attack = profile["attack_rows"] > 0
        has_control = profile["control_rows"] > 0
        if has_attack and has_control:
            selected_profile["mixed_items"] += 1
        elif has_attack:
            selected_profile["attack_items"] += 1
        elif has_control:
            selected_profile["control_items"] += 1

    if (
        selected_profile["mixed_items"] == 0
        and (
            selected_profile["attack_items"] == 0
            or selected_profile["control_items"] == 0
        )
    ):
        raise RuntimeError(
            "internal compact balanced-index failure: "
            f"{selected_profile}"
        )

    return selected, {
        "items_scanned": scanned,
        "items_selected": len(selected),
        "selected_indices": selected,
        **selected_profile,
    }


def flatten_pair_aligned_batch(
    batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(batch, dict):
        raise TypeError("DataLoader batch is not a dictionary")

    x = batch_tensor(batch, ("x", "features", "input"))
    target = batch_tensor(
        batch,
        ("y_source", "source", "source_target"),
    )

    if not torch.is_tensor(x) or not torch.is_tensor(target):
        raise TypeError("x and y_source must be tensors after collation")
    if target.shape[-1] != 16:
        raise RuntimeError(
            f"y_source last dimension is {target.shape[-1]}, expected 16"
        )

    # Standard item-wise dataset: [batch,16].
    if target.ndim == 2:
        if x.ndim < 3 or x.shape[0] != target.shape[0]:
            raise RuntimeError(
                "standard batch x/y_source leading dimensions disagree: "
                f"x={tuple(x.shape)}, y={tuple(target.shape)}"
            )
        flattened = dict(batch)
        prefix_shape = [int(target.shape[0])]
        pair_aligned = False

    # Pair/block dataset: e.g. y_source [blocks,2,16] and
    # x [blocks,2,16,time,features]. Flatten every tensor that shares the
    # same leading block/pair prefix.
    elif target.ndim >= 3:
        prefix = tuple(int(value) for value in target.shape[:-1])
        prefix_rank = len(prefix)
        if tuple(int(value) for value in x.shape[:prefix_rank]) != prefix:
            raise RuntimeError(
                "pair-aligned x/y_source prefixes disagree: "
                f"x={tuple(x.shape)}, y={tuple(target.shape)}"
            )

        flattened_count = int(math.prod(prefix))
        flattened = {}
        for key, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.ndim >= prefix_rank
                and tuple(int(v) for v in value.shape[:prefix_rank]) == prefix
            ):
                flattened[key] = value.reshape(
                    flattened_count,
                    *value.shape[prefix_rank:],
                )
            else:
                flattened[key] = value

        prefix_shape = list(prefix)
        pair_aligned = True
    else:
        raise RuntimeError(
            f"unsupported y_source batch shape: {tuple(target.shape)}"
        )

    flat_x = batch_tensor(flattened, ("x", "features", "input"))
    flat_target = batch_tensor(
        flattened,
        ("y_source", "source", "source_target"),
    )

    if flat_target.ndim != 2 or list(flat_target.shape[1:]) != [16]:
        raise RuntimeError(
            "flattened y_source must be [items,16], got "
            f"{tuple(flat_target.shape)}"
        )
    if flat_x.shape[0] != flat_target.shape[0]:
        raise RuntimeError(
            "flattened x/y_source item counts disagree: "
            f"x={tuple(flat_x.shape)}, y={tuple(flat_target.shape)}"
        )
    if flat_x.ndim < 3 or flat_x.shape[1] != 16:
        raise RuntimeError(
            "flattened x must have router dimension 16 at axis 1, got "
            f"{tuple(flat_x.shape)}"
        )

    profile = source_row_profile(flat_target)
    if profile["attack_rows"] == 0 or profile["control_rows"] == 0:
        raise RuntimeError(
            "smoke batch does not contain both attack and control rows: "
            f"{profile}"
        )

    return flattened, {
        "pair_aligned_input": pair_aligned,
        "original_prefix_shape": prefix_shape,
        "flattened_x_shape": list(flat_x.shape),
        "flattened_y_source_shape": list(flat_target.shape),
        **profile,
    }


def tensor_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def module_contains_probability_activation(module: nn.Module) -> bool:
    forbidden = (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)
    return any(
        isinstance(child, forbidden)
        for child in module.modules()
    )


def gradient_groups(model: nn.Module) -> dict[str, Any]:
    groups = {
        "temporal_backbone": [],
        "graph1": [],
        "graph2": [],
        "source_head": [],
        "other": [],
    }

    cutpoint_prefix = (
        f"adapter.b3.{model.adapter.cutpoint_name}."
        if model.adapter.cutpoint_name
        else None
    )

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("graph1."):
            group = "graph1"
        elif name.startswith("graph2."):
            group = "graph2"
        elif name.startswith("source_head."):
            group = "source_head"
        elif name.startswith("adapter.b3."):
            # The original B3 task head executes only to expose its input.
            # Exclude that original head from the temporal-backbone group.
            if cutpoint_prefix and name.startswith(cutpoint_prefix):
                group = "other"
            else:
                group = "temporal_backbone"
        else:
            group = "other"

        grad = parameter.grad
        groups[group].append(
            {
                "name": name,
                "has_gradient": grad is not None,
                "finite": (
                    bool(torch.isfinite(grad).all().item())
                    if grad is not None
                    else None
                ),
                "absolute_sum": (
                    float(grad.detach().abs().sum().cpu())
                    if grad is not None
                    else 0.0
                ),
            }
        )

    summary = {}
    for group, rows in groups.items():
        with_gradient = [
            row for row in rows if row["has_gradient"]
        ]
        summary[group] = {
            "parameter_tensors": len(rows),
            "with_gradient": len(with_gradient),
            "all_gradient_tensors_finite": all(
                row["finite"] for row in with_gradient
            ) if with_gradient else False,
            "total_absolute_gradient": float(
                sum(row["absolute_sum"] for row in rows)
            ),
            "gradient_parameter_names": [
                row["name"] for row in with_gradient
            ],
        }
    return summary


def assert_required_gradient_groups(
    operator: str,
    summary: dict[str, Any],
) -> None:
    required = ["temporal_backbone", "source_head"]
    if operator != "conv1d":
        required.extend(["graph1", "graph2"])

    for group in required:
        details = summary[group]
        if details["with_gradient"] == 0:
            raise RuntimeError(
                f"{operator}: required group {group} received no gradient"
            )
        if not details["all_gradient_tensors_finite"]:
            raise RuntimeError(
                f"{operator}: non-finite gradient in group {group}"
            )
        # A required group must be connected to the loss graph. A finite
        # zero gradient is still possible for a valid ReLU/dropout state in
        # a one-batch smoke, so connectivity is checked by grad presence.
        # The copied source head must additionally receive a non-zero signal.
        if (
            group == "source_head"
            and details["total_absolute_gradient"] <= 0.0
        ):
            raise RuntimeError(
                f"{operator}: copied source head received zero gradient"
            )


def find_model_classes(module) -> list[type]:
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, nn.Module):
                classes.append(obj)
        except TypeError:
            pass
    return classes


def instantiate_b3(module):
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

    attempts: list[dict[str, Any]] = []
    exact_matches: list[tuple[nn.Module, type, dict[str, Any]]] = []
    seen_candidates: set[tuple[str, str]] = set()

    for cls in find_model_classes(module):
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
            key = (cls.__name__, json.dumps(candidate, sort_keys=True))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            try:
                model = cls(**candidate)
                parameter_count = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                attempts.append(
                    {
                        "class": cls.__name__,
                        "kwargs": candidate,
                        "parameter_count": parameter_count,
                    }
                )
                if parameter_count == EXPECTED_B3_PARAMETER_COUNT:
                    exact_matches.append((model, cls, candidate))
            except Exception as exc:
                attempts.append(
                    {
                        "class": cls.__name__,
                        "kwargs": candidate,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    if len(exact_matches) != 1:
        raise RuntimeError(
            "expected exactly one 43,273-parameter B3 model, found "
            f"{len(exact_matches)}; attempts={attempts}"
        )

    model, cls, candidate = exact_matches[0]
    return model, cls, candidate, attempts


def module_has_parameters(module: nn.Module) -> bool:
    return any(True for _ in module.parameters(recurse=True))


def select_source_cutpoint(model: nn.Module):
    named = list(model.named_modules())
    candidates = []

    for name, module in named:
        if not name:
            continue
        lowered = name.lower()
        if "source" not in lowered and "attacker" not in lowered:
            continue
        if not module_has_parameters(module):
            continue

        score = 0
        if lowered.endswith("source_head"):
            score += 100
        if lowered == "source_head":
            score += 100
        if "source_head" in lowered:
            score += 50
        if lowered.endswith("source"):
            score += 25
        if isinstance(module, nn.Sequential):
            score += 20

        # Prefer the outer source head over a child linear layer because its
        # input is the shared node embedding before task-specific transforms.
        depth = name.count(".")
        score -= depth * 5
        candidates.append((score, -len(name), name, module))

    if not candidates:
        raise RuntimeError(
            "no parameterized module containing 'source' or 'attacker' "
            "was found in the B3 model"
        )

    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, _, name, module = candidates[0]
    return name, module, [
        {
            "name": candidate_name,
            "class": candidate_module.__class__.__name__,
            "score": candidate_score,
        }
        for (
            candidate_score,
            _,
            candidate_name,
            candidate_module,
        ) in candidates
    ]


def batch_tensor(batch: dict[str, Any], names: tuple[str, ...]):
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f"none of the required batch keys exist: {names}")


def call_b3(model: nn.Module, batch: dict[str, Any]):
    x = batch_tensor(batch, ("x", "features", "input"))
    signature = inspect.signature(model.forward)
    parameter_names = [
        name
        for name in signature.parameters
        if name != "self"
    ]

    attempts = []

    variants = [
        ("positional_x", (x,), {}),
        ("keyword_x", (), {"x": x}),
        ("batch_dict", (batch,), {}),
    ]

    kwargs_from_batch = {
        name: batch[name]
        for name in parameter_names
        if name in batch
    }
    if kwargs_from_batch:
        variants.append(
            ("signature_batch_kwargs", (), kwargs_from_batch)
        )

    for variant_name, args, kwargs in variants:
        try:
            output = model(*args, **kwargs)
            return output, variant_name, attempts
        except Exception as exc:
            attempts.append(
                {
                    "variant": variant_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    raise RuntimeError(
        "all B3 forward call variants failed: "
        f"signature={signature}, attempts={attempts}"
    )


def normalize_node_embedding(
    value: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(
            f"captured source-head input is {type(value).__name__}, "
            "not a tensor"
        )

    if value.ndim == 3:
        if value.shape[0] == batch_size and value.shape[1] == num_nodes:
            return value
        if value.shape[0] == batch_size and value.shape[2] == num_nodes:
            return value.transpose(1, 2).contiguous()

    if value.ndim == 2:
        if value.shape[0] == batch_size * num_nodes:
            return value.reshape(batch_size, num_nodes, value.shape[-1])
        if batch_size == 1 and value.shape[0] == num_nodes:
            return value.unsqueeze(0)

    raise RuntimeError(
        "unsupported source-head input shape: "
        f"{tuple(value.shape)} for batch={batch_size}, nodes={num_nodes}"
    )


class B3NodeEmbeddingAdapter(nn.Module):
    def __init__(self, b3: nn.Module):
        super().__init__()
        self.b3 = b3
        (
            self.cutpoint_name,
            self.cutpoint_module,
            self.cutpoint_candidates,
        ) = select_source_cutpoint(self.b3)
        self.forward_variant: str | None = None
        self.last_capture_shape: list[int] | None = None

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x = batch_tensor(batch, ("x", "features", "input"))
        if x.ndim < 3:
            raise RuntimeError(f"unexpected x shape: {tuple(x.shape)}")
        batch_size = int(x.shape[0])
        num_nodes = 16

        captured: list[Any] = []

        def hook(_module, args):
            if not args:
                raise RuntimeError(
                    "source cutpoint pre-hook received no positional input"
                )
            captured.append(args[0])

        handle = self.cutpoint_module.register_forward_pre_hook(hook)
        try:
            _, variant, _ = call_b3(self.b3, batch)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(
                f"expected one source cutpoint capture, got {len(captured)}"
            )

        embedding = normalize_node_embedding(
            captured[0],
            batch_size=batch_size,
            num_nodes=num_nodes,
        )
        self.forward_variant = variant
        self.last_capture_shape = list(embedding.shape)
        return embedding


def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    base = base_edge_index.to(device=device, dtype=torch.long)
    pieces = [
        base + graph_index * num_nodes
        for graph_index in range(batch_size)
    ]
    result = torch.cat(pieces, dim=1)

    # Prove that no edge crosses a graph boundary.
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
        raise RuntimeError("batched edge_index contains cross-graph edges")
    return result


class SourceOnlyBaseline(nn.Module):
    def __init__(
        self,
        b3: nn.Module,
        operator: str,
        embedding_dim: int,
        base_edge_index: torch.Tensor,
    ):
        super().__init__()
        self.adapter = B3NodeEmbeddingAdapter(b3)
        self.operator = operator
        self.base_edge_index = base_edge_index

        # Clone the exact frozen B3 source-head architecture and initial
        # weights. This makes the source head identical across all operators
        # for the same seed and avoids inventing a scientifically different
        # one-layer classifier.
        self.source_head = copy.deepcopy(
            self.adapter.cutpoint_module
        )
        if module_contains_probability_activation(self.source_head):
            raise RuntimeError(
                "B3 source head contains Sigmoid/Softmax; "
                "BCEWithLogitsLoss requires raw logits"
            )

        if operator == "conv1d":
            self.graph1 = None
            self.graph2 = None
        elif operator == "gcnconv":
            self.graph1 = GCNConv(
                embedding_dim,
                embedding_dim,
                add_self_loops=True,
                normalize=True,
            )
            self.graph2 = GCNConv(
                embedding_dim,
                embedding_dim,
                add_self_loops=True,
                normalize=True,
            )
        elif operator == "graphconv":
            self.graph1 = GraphConv(
                embedding_dim,
                embedding_dim,
                aggr="add",
            )
            self.graph2 = GraphConv(
                embedding_dim,
                embedding_dim,
                aggr="add",
            )
        elif operator == "gatconv":
            self.graph1 = GATConv(
                embedding_dim,
                embedding_dim,
                heads=1,
                concat=False,
                add_self_loops=True,
            )
            self.graph2 = GATConv(
                embedding_dim,
                embedding_dim,
                heads=1,
                concat=False,
                add_self_loops=True,
            )
        else:
            raise ValueError(f"unknown operator: {operator}")

        self.activation = nn.ReLU()

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        node_embedding = self.adapter(batch)
        batch_size, num_nodes, embedding_dim = node_embedding.shape

        if self.operator != "conv1d":
            flat = node_embedding.reshape(
                batch_size * num_nodes,
                embedding_dim,
            )
            edge_index = batched_edge_index(
                self.base_edge_index,
                batch_size=batch_size,
                num_nodes=num_nodes,
                device=flat.device,
            )
            flat = self.activation(self.graph1(flat, edge_index))
            flat = self.activation(self.graph2(flat, edge_index))
            node_embedding = flat.reshape(
                batch_size,
                num_nodes,
                embedding_dim,
            )

        output = self.source_head(node_embedding)
        if output.ndim == 3 and output.shape[-1] == 1:
            return output.squeeze(-1)
        if output.ndim == 2:
            return output
        raise RuntimeError(
            "source head must return [batch,16] or [batch,16,1], got "
            f"{tuple(output.shape)}"
        )


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def gradient_summary(model: nn.Module) -> dict[str, Any]:
    trainable = 0
    with_gradient = 0
    finite_gradient = 0
    total_abs_gradient = 0.0

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        trainable += 1
        if parameter.grad is None:
            continue
        with_gradient += 1
        if torch.isfinite(parameter.grad).all():
            finite_gradient += 1
        total_abs_gradient += float(
            parameter.grad.detach().abs().sum().cpu()
        )

    return {
        "trainable_parameter_tensors": trainable,
        "parameter_tensors_with_gradient": with_gradient,
        "parameter_tensors_with_finite_gradient": finite_gradient,
        "total_absolute_gradient": total_abs_gradient,
    }


def smoke_one_operator(
    operator: str,
    model_module,
    train_batch_cpu: dict[str, Any],
    validation_batch_cpu: dict[str, Any],
    base_edge_index: torch.Tensor,
    device: torch.device,
    embedding_dim: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    b3, b3_class, b3_kwargs, b3_attempts = instantiate_b3(
        model_module
    )

    b3_parameter_count = sum(
        parameter.numel()
        for parameter in b3.parameters()
    )

    model = SourceOnlyBaseline(
        b3=b3,
        operator=operator,
        embedding_dim=embedding_dim,
        base_edge_index=base_edge_index,
    ).to(device)

    initial_b3_state_sha256 = tensor_state_sha256(
        model.adapter.b3
    )
    initial_source_head_state_sha256 = tensor_state_sha256(
        model.source_head
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_batch = move_batch_to_device(
        train_batch_cpu,
        device,
    )
    validation_batch = move_batch_to_device(
        validation_batch_cpu,
        device,
    )

    target = batch_tensor(
        train_batch,
        ("y_source", "source", "source_target"),
    ).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_batch)

    if logits.shape != target.shape:
        raise RuntimeError(
            f"{operator}: logits shape {tuple(logits.shape)} "
            f"does not match source target {tuple(target.shape)}"
        )

    loss = criterion(logits, target)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{operator}: non-finite training loss")

    loss.backward()
    gradients = gradient_summary(model)
    grouped_gradients = gradient_groups(model)
    assert_required_gradient_groups(
        operator,
        grouped_gradients,
    )

    if gradients["parameter_tensors_with_gradient"] == 0:
        raise RuntimeError(f"{operator}: no parameter received a gradient")
    if (
        gradients["parameter_tensors_with_gradient"]
        != gradients["parameter_tensors_with_finite_gradient"]
    ):
        raise RuntimeError(f"{operator}: non-finite gradients detected")

    optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_batch)

    validation_target = batch_tensor(
        validation_batch,
        ("y_source", "source", "source_target"),
    ).float()
    validation_loss = criterion(
        validation_logits,
        validation_target,
    )

    if not torch.isfinite(validation_loss):
        raise RuntimeError(
            f"{operator}: non-finite validation loss"
        )

    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "allocated_bytes": int(
                torch.cuda.memory_allocated(device)
            ),
            "reserved_bytes": int(
                torch.cuda.memory_reserved(device)
            ),
            "max_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }

    return {
        "operator": operator,
        "status": "PASS",
        "seed": seed,
        "b3_class": b3_class.__name__,
        "b3_kwargs": b3_kwargs,
        "b3_failed_instantiation_attempts": b3_attempts,
        "b3_parameter_count": b3_parameter_count,
        "source_cutpoint_name": model.adapter.cutpoint_name,
        "source_cutpoint_class": (
            model.adapter.cutpoint_module.__class__.__name__
        ),
        "source_cutpoint_candidates": (
            model.adapter.cutpoint_candidates
        ),
        "source_head_clone_class": (
            model.source_head.__class__.__name__
        ),
        "initial_b3_state_sha256": initial_b3_state_sha256,
        "initial_source_head_state_sha256": (
            initial_source_head_state_sha256
        ),
        "b3_forward_variant": model.adapter.forward_variant,
        "node_embedding_shape": (
            model.adapter.last_capture_shape
        ),
        "embedding_dim": embedding_dim,
        "train_logits_shape": list(logits.shape),
        "validation_logits_shape": list(
            validation_logits.shape
        ),
        "train_loss_before_step": float(
            loss.detach().cpu()
        ),
        "validation_loss_after_step": float(
            validation_loss.detach().cpu()
        ),
        "gradient_summary": gradients,
        "gradient_groups": grouped_gradients,
        "total_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "cuda_memory": cuda_memory,
    }


def infer_embedding_dim(
    model_module,
    batch_cpu: dict[str, Any],
    device: torch.device,
):
    set_seed(107)
    b3, b3_class, b3_kwargs, _ = instantiate_b3(model_module)
    parameter_count = sum(
        parameter.numel()
        for parameter in b3.parameters()
    )
    if parameter_count != EXPECTED_B3_PARAMETER_COUNT:
        raise RuntimeError(
            f"B3 parameter count {parameter_count} does not match "
            f"frozen value {EXPECTED_B3_PARAMETER_COUNT}"
        )

    adapter = B3NodeEmbeddingAdapter(b3).to(device)
    batch = move_batch_to_device(batch_cpu, device)
    adapter.eval()
    with torch.no_grad():
        embedding = adapter(batch)

    return {
        "embedding_dim": int(embedding.shape[-1]),
        "embedding_shape": list(embedding.shape),
        "cutpoint_name": adapter.cutpoint_name,
        "cutpoint_class": adapter.cutpoint_module.__class__.__name__,
        "cutpoint_candidates": adapter.cutpoint_candidates,
        "forward_variant": adapter.forward_variant,
        "b3_class": b3_class.__name__,
        "b3_kwargs": b3_kwargs,
        "b3_parameter_count": parameter_count,
    }



def validate_flat_batch(
    batch: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    x = batch_tensor(batch, ("x", "features", "input"))
    y_source = batch_tensor(
        batch,
        ("y_source", "source", "source_target"),
    )

    if not torch.is_tensor(x) or not torch.is_tensor(y_source):
        raise TypeError(f"{split_name}: x/y_source are not tensors")
    if x.ndim != 4:
        raise RuntimeError(
            f"{split_name}: x must be [items,16,58,32], got {tuple(x.shape)}"
        )
    if tuple(x.shape[1:]) != (16, 58, 32):
        raise RuntimeError(
            f"{split_name}: x tail must be [16,58,32], got {tuple(x.shape[1:])}"
        )
    if y_source.ndim != 2 or y_source.shape[1] != 16:
        raise RuntimeError(
            f"{split_name}: y_source must be [items,16], got "
            f"{tuple(y_source.shape)}"
        )
    if x.shape[0] != y_source.shape[0]:
        raise RuntimeError(
            f"{split_name}: x/y_source item counts disagree: "
            f"{x.shape[0]} vs {y_source.shape[0]}"
        )
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{split_name}: x contains non-finite values")
    if not torch.isfinite(y_source.float()).all():
        raise RuntimeError(
            f"{split_name}: y_source contains non-finite values"
        )

    unique_targets = torch.unique(y_source.detach().cpu())
    allowed = {0, 1, False, True}
    values = {value.item() for value in unique_targets}
    if not values.issubset(allowed):
        raise RuntimeError(
            f"{split_name}: y_source is not binary; values={sorted(values)}"
        )

    profile = source_row_profile(y_source)
    if profile["attack_rows"] == 0 or profile["control_rows"] == 0:
        raise RuntimeError(
            f"{split_name}: smoke batch lacks attack/control coverage: "
            f"{profile}"
        )

    return {
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_source_shape": list(y_source.shape),
        "y_source_dtype": str(y_source.dtype),
        "binary_values": sorted(int(value) for value in values),
        **profile,
    }


def normalize_source_output(
    value: Any,
    batch_size: int,
    num_nodes: int = 16,
) -> torch.Tensor:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(
            f"source-head output is {type(value).__name__}, not a tensor"
        )

    if value.ndim == 3 and value.shape == (batch_size, num_nodes, 1):
        return value.squeeze(-1)
    if value.ndim == 2 and value.shape == (batch_size, num_nodes):
        return value
    if value.ndim == 2 and value.shape == (batch_size * num_nodes, 1):
        return value.reshape(batch_size, num_nodes)
    if value.ndim == 1 and value.shape[0] == batch_size * num_nodes:
        return value.reshape(batch_size, num_nodes)

    raise RuntimeError(
        "unsupported source-head output shape: "
        f"{tuple(value.shape)} for batch={batch_size}, nodes={num_nodes}"
    )


def probe_source_head_equivalence(
    model_module,
    batch_cpu: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    set_seed(107)
    b3, b3_class, b3_kwargs, attempts = instantiate_b3(model_module)
    adapter = B3NodeEmbeddingAdapter(b3).to(device)
    batch = move_batch_to_device(batch_cpu, device)
    x = batch_tensor(batch, ("x", "features", "input"))
    batch_size = int(x.shape[0])

    captured_inputs: list[Any] = []
    captured_outputs: list[Any] = []

    def pre_hook(_module, args):
        if not args:
            raise RuntimeError("source head pre-hook received no input")
        captured_inputs.append(args[0])

    def post_hook(_module, _args, output):
        captured_outputs.append(output)

    pre_handle = adapter.cutpoint_module.register_forward_pre_hook(pre_hook)
    post_handle = adapter.cutpoint_module.register_forward_hook(post_hook)
    adapter.b3.eval()
    try:
        with torch.no_grad():
            _, variant, forward_attempts = call_b3(adapter.b3, batch)
    finally:
        pre_handle.remove()
        post_handle.remove()

    if len(captured_inputs) != 1 or len(captured_outputs) != 1:
        raise RuntimeError(
            "source head must execute exactly once during B3 forward: "
            f"inputs={len(captured_inputs)}, outputs={len(captured_outputs)}"
        )

    embedding = normalize_node_embedding(
        captured_inputs[0],
        batch_size=batch_size,
        num_nodes=16,
    )
    original_logits = normalize_source_output(
        captured_outputs[0],
        batch_size=batch_size,
    )

    cloned_head = copy.deepcopy(adapter.cutpoint_module).to(device)
    cloned_head.eval()
    if module_contains_probability_activation(cloned_head):
        raise RuntimeError(
            "B3 source head contains a probability activation; "
            "raw logits are required"
        )
    with torch.no_grad():
        cloned_logits = normalize_source_output(
            cloned_head(embedding),
            batch_size=batch_size,
        )

    if not torch.isfinite(original_logits).all():
        raise RuntimeError("original B3 source logits are non-finite")
    if not torch.isfinite(cloned_logits).all():
        raise RuntimeError("cloned source-head logits are non-finite")

    max_abs_error = float(
        (original_logits - cloned_logits).abs().max().detach().cpu()
    )
    if max_abs_error > 1.0e-7:
        raise RuntimeError(
            "captured embedding + cloned source head does not reproduce "
            f"the B3 source head; max_abs_error={max_abs_error}"
        )

    return {
        "b3_class": b3_class.__name__,
        "b3_kwargs": b3_kwargs,
        "b3_instantiation_attempts": attempts,
        "b3_parameter_count": sum(
            parameter.numel() for parameter in b3.parameters()
        ),
        "cutpoint_name": adapter.cutpoint_name,
        "cutpoint_class": adapter.cutpoint_module.__class__.__name__,
        "cutpoint_candidates": adapter.cutpoint_candidates,
        "forward_variant": variant,
        "forward_attempts_before_success": forward_attempts,
        "raw_cutpoint_input_shape": list(captured_inputs[0].shape),
        "normalized_embedding_shape": list(embedding.shape),
        "embedding_dim": int(embedding.shape[-1]),
        "original_source_logits_shape": list(original_logits.shape),
        "cloned_source_logits_shape": list(cloned_logits.shape),
        "clone_equivalence_max_abs_error": max_abs_error,
        "original_source_head_state_sha256": tensor_state_sha256(
            adapter.cutpoint_module
        ),
        "cloned_source_head_state_sha256": tensor_state_sha256(cloned_head),
    }


def validate_edge_index(edge_index: torch.Tensor) -> dict[str, Any]:
    edge = edge_index.detach().cpu().to(dtype=torch.long).contiguous()
    if list(edge.shape) != [2, 48]:
        raise RuntimeError(
            f"edge_index shape is {list(edge.shape)}, expected [2,48]"
        )
    if int(edge.min()) != 0 or int(edge.max()) != 15:
        raise RuntimeError("edge_index router range is not exactly 0..15")
    if int((edge[0] == edge[1]).sum()) != 0:
        raise RuntimeError("physical edge_index contains self-loops")

    edges = [tuple(map(int, item)) for item in edge.t().tolist()]
    edge_set = set(edges)
    if len(edge_set) != 48:
        raise RuntimeError("edge_index does not contain 48 unique edges")
    for src, dst in edge_set:
        src_row, src_col = divmod(src, 4)
        dst_row, dst_col = divmod(dst, 4)
        if abs(src_row - dst_row) + abs(src_col - dst_col) != 1:
            raise RuntimeError(f"non-physical mesh edge: {src}->{dst}")
        if (dst, src) not in edge_set:
            raise RuntimeError(f"missing reverse edge for {src}->{dst}")

    return {
        "shape": [2, 48],
        "dtype": str(edge.dtype),
        "minimum_router": 0,
        "maximum_router": 15,
        "directed_edges": 48,
        "undirected_links": 24,
        "self_loops": 0,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def run_smoke(args: argparse.Namespace, output_dir: Path) -> int:
    failures: list[str] = []
    warnings: list[str] = []
    results: dict[str, Any] = {}
    resolved_contract: dict[str, Any] = {}

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    r1_dir = args.r1_dir.expanduser().resolve()
    topology_dir = args.topology_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    loader_path = args.loader.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    g0_lock_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    r1_report_path = (
        r1_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT.json"
    )
    topology_report_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json"
    )
    topology_lock_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json"
    )
    topology_complete_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_COMPLETE"
    )
    edge_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy"
    )

    if PYG_IMPORT_ERROR is not None:
        failures.append(
            "PyTorch Geometric graph operators are unavailable: "
            f"{PYG_IMPORT_ERROR}"
        )

    prerequisites = (
        g0_lock_path,
        r1_report_path,
        topology_report_path,
        topology_lock_path,
        topology_complete_path,
        edge_path,
        loader_path,
        model_path,
    )
    for path in prerequisites:
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if not data_root.is_dir():
        failures.append(f"missing data root: {data_root}")

    g0_lock: dict[str, Any] = {}
    r1_report: dict[str, Any] = {}
    topology_report: dict[str, Any] = {}
    topology_lock: dict[str, Any] = {}
    canonical_topology: dict[str, Any] = {}

    if not failures:
        try:
            g0_lock = load_json(g0_lock_path)
            r1_report = load_json(r1_report_path)
            topology_report = load_json(topology_report_path)
            topology_lock = load_json(topology_lock_path)

            if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
                failures.append("G0 protocol SHA changed")
            if topology_report.get("status") != "COMPLETE":
                failures.append("topology report is not COMPLETE")
            if (
                topology_report.get("decision")
                != "FREEZE_CANONICAL_ROW_MAJOR_4X4_TOPOLOGY_AND_AUTHORIZE_G1"
            ):
                failures.append("topology decision changed")

            canonical_topology = topology_report.get("canonical_contract", {})
            if not isinstance(canonical_topology, dict):
                failures.append("canonical_contract is not a JSON object")
                canonical_topology = {}

            report_sha = sha256_file(topology_report_path)
            edge_sha = sha256_file(edge_path)
            if topology_lock.get("report_sha256") != report_sha:
                failures.append("topology report SHA disagrees with lock")
            if topology_lock.get("contract_file_sha256") != report_sha:
                failures.append(
                    "topology contract-file SHA disagrees with lock"
                )
            if (
                topology_lock.get("contract_sha256")
                != canonical_topology.get("contract_sha256")
            ):
                failures.append("canonical topology SHA disagrees with lock")
            if topology_lock.get("edge_index_sha256") != edge_sha:
                failures.append("edge_index SHA disagrees with lock")
            if edge_sha != EXPECTED_EDGE_ARTIFACT_SHA:
                failures.append("canonical edge_index artifact changed")

            edge_contract = (
                canonical_topology
                .get("artifacts", {})
                .get("edge_index", {})
            )
            physical = canonical_topology.get("physical_routing_graph", {})
            expected_semantics = {
                "topology": canonical_topology.get("topology"),
                "num_nodes": canonical_topology.get("num_nodes"),
                "router_numbering": canonical_topology.get(
                    "router_numbering"
                ),
                "edge_shape": edge_contract.get("shape"),
                "edge_sha": edge_contract.get("sha256"),
                "directed_edges": physical.get("directed_edge_count"),
                "undirected_links": physical.get("undirected_link_count"),
                "physical_self_loops": physical.get(
                    "physical_self_loops"
                ),
            }
            if expected_semantics != {
                "topology": "4x4_2D_MESH",
                "num_nodes": 16,
                "router_numbering": "row_major",
                "edge_shape": [2, 48],
                "edge_sha": edge_sha,
                "directed_edges": 48,
                "undirected_links": 24,
                "physical_self_loops": False,
            }:
                failures.append(
                    "canonical topology semantics changed: "
                    f"{expected_semantics}"
                )
        except Exception as exc:
            failures.append(
                "prerequisite validation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    train_dataset = None
    validation_dataset = None
    train_selection = None
    validation_selection = None
    train_batch_contract = None
    validation_batch_contract = None
    dataset_class = None
    train_manifest = None
    validation_manifest = None

    if not failures:
        try:
            selected = (
                r1_report
                .get("manifest_discovery", {})
                .get("selected", {})
            )
            train_manifest_text = selected.get("train_manifest")
            validation_manifest_text = selected.get("validation_manifest")
            dataset_class_name = (
                r1_report.get("loader", {}).get("selected_class")
            )
            if not train_manifest_text or not validation_manifest_text:
                raise RuntimeError(
                    "G1A-R1 report does not contain resolved manifests"
                )

            train_manifest = Path(train_manifest_text).expanduser().resolve()
            validation_manifest = (
                Path(validation_manifest_text).expanduser().resolve()
            )
            if not train_manifest.is_file():
                raise RuntimeError(
                    f"train manifest missing: {train_manifest}"
                )
            if not validation_manifest.is_file():
                raise RuntimeError(
                    f"validation manifest missing: {validation_manifest}"
                )

            loader_module = import_module_from_path(
                "v5_p2_g1_clean_loader",
                loader_path,
            )
            dataset_class = find_dataset_class(
                loader_module,
                dataset_class_name,
            )
            train_dataset = dataset_class(
                **build_dataset_kwargs(
                    dataset_class,
                    data_root,
                    "train",
                    train_manifest,
                )
            )
            validation_dataset = dataset_class(
                **build_dataset_kwargs(
                    dataset_class,
                    data_root,
                    "validation",
                    validation_manifest,
                )
            )

            requested_items = max(2, args.batch_size)
            train_indices, train_selection = select_balanced_smoke_indices(
                train_dataset,
                requested_items=requested_items,
                max_scan=args.max_scan,
            )
            validation_indices, validation_selection = (
                select_balanced_smoke_indices(
                    validation_dataset,
                    requested_items=requested_items,
                    max_scan=args.max_scan,
                )
            )

            train_subset = Subset(train_dataset, train_indices)
            validation_subset = Subset(
                validation_dataset,
                validation_indices,
            )
            train_loader = DataLoader(
                train_subset,
                batch_size=len(train_subset),
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            validation_loader = DataLoader(
                validation_subset,
                batch_size=len(validation_subset),
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            train_batch, train_flatten = flatten_pair_aligned_batch(
                next(iter(train_loader))
            )
            validation_batch, validation_flatten = (
                flatten_pair_aligned_batch(
                    next(iter(validation_loader))
                )
            )
            train_validation = validate_flat_batch(train_batch, "train")
            validation_validation = validate_flat_batch(
                validation_batch,
                "validation",
            )
            train_batch_contract = {
                **train_flatten,
                **train_validation,
            }
            validation_batch_contract = {
                **validation_flatten,
                **validation_validation,
            }
        except Exception as exc:
            failures.append(
                "dataset/batch preparation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    edge_index = None
    edge_contract_runtime = None
    if not failures:
        try:
            edge_index = torch.from_numpy(
                np.load(edge_path, allow_pickle=False)
            ).to(dtype=torch.long)
            edge_contract_runtime = validate_edge_index(edge_index)
        except Exception as exc:
            failures.append(
                "edge_index validation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    device = None
    model_module = None
    interface_probe = None

    if not failures:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            model_module = import_module_from_path(
                "v5_p2_g1_clean_model",
                model_path,
            )
            interface_probe = probe_source_head_equivalence(
                model_module,
                train_batch,
                device,
            )
            embedding_dim = int(interface_probe["embedding_dim"])

            for operator in (
                "conv1d",
                "gcnconv",
                "graphconv",
                "gatconv",
            ):
                results[operator] = smoke_one_operator(
                    operator=operator,
                    model_module=model_module,
                    train_batch_cpu=train_batch,
                    validation_batch_cpu=validation_batch,
                    base_edge_index=edge_index,
                    device=device,
                    embedding_dim=embedding_dim,
                    seed=args.seed,
                )

            shared_fields = {
                "source_cutpoint_name": {
                    value["source_cutpoint_name"]
                    for value in results.values()
                },
                "embedding_dim": {
                    value["embedding_dim"] for value in results.values()
                },
                "b3_forward_variant": {
                    value["b3_forward_variant"]
                    for value in results.values()
                },
                "initial_b3_state_sha256": {
                    value["initial_b3_state_sha256"]
                    for value in results.values()
                },
                "initial_source_head_state_sha256": {
                    value["initial_source_head_state_sha256"]
                    for value in results.values()
                },
            }
            for field, values in shared_fields.items():
                if len(values) != 1:
                    failures.append(
                        f"operators disagree on {field}: {sorted(values)}"
                    )

            resolved_core = {
                "name": "V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CONTRACT",
                "version": 1,
                "dataset_class": dataset_class.__name__,
                "train_manifest": str(train_manifest),
                "validation_manifest": str(validation_manifest),
                "input_shape": ["items", 16, 58, 32],
                "source_target_shape": ["items", 16],
                "source_target_semantics": (
                    "16 independent binary attacker/source labels"
                ),
                "temporal_backbone": interface_probe["b3_class"],
                "temporal_backbone_parameters": (
                    interface_probe["b3_parameter_count"]
                ),
                "source_embedding_cutpoint": (
                    interface_probe["cutpoint_name"]
                ),
                "source_embedding_dim": embedding_dim,
                "source_head_clone_equivalence_max_abs_error": (
                    interface_probe["clone_equivalence_max_abs_error"]
                ),
                "canonical_edge_index_sha256": sha256_file(edge_path),
                "operators": [
                    "conv1d",
                    "gcnconv",
                    "graphconv",
                    "gatconv",
                ],
                "graph_layers": 2,
                "graph_width": embedding_dim,
                "initial_seeds": [107, 117, 127],
                "execution_policy": "serial one-model-at-a-time",
                "production_constraints": {
                    "hook_based_full_b3_wrapper_allowed": False,
                    "clean_temporal_encoder_extraction_required": True,
                    "unused_multitask_heads_allowed": False,
                    "clean_encoder_equivalence_required": True,
                },
                "scientific_training_authorized": False,
                "next_stage": (
                    "V5_P2_G1C_SOURCE_ONLY_GRAPH_TRAINING_MATRIX"
                ),
            }
            resolved_contract = {
                **resolved_core,
                "contract_sha256": canonical_sha256(resolved_core),
            }
        except Exception as exc:
            failures.append(
                "model/operator smoke failed: "
                f"{type(exc).__name__}: {exc}"
            )

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "decision": (
            "AUTHORIZE_CLEAN_G1C_SOURCE_ONLY_TRAINING_IMPLEMENTATION"
            if not failures
            else "HOLD_G1C_PENDING_SMOKE_FIX"
        ),
        "historical_smoke_outputs_touched": False,
        "device": str(device) if device is not None else None,
        "torch_version": torch.__version__,
        "dataset": {
            "class": dataset_class.__name__ if dataset_class else None,
            "train_items": len(train_dataset) if train_dataset else None,
            "validation_items": (
                len(validation_dataset) if validation_dataset else None
            ),
            "train_selection": train_selection,
            "validation_selection": validation_selection,
            "train_batch": train_batch_contract,
            "validation_batch": validation_batch_contract,
        },
        "topology": {
            "contract_sha256": canonical_topology.get("contract_sha256"),
            "edge_index_sha256": (
                sha256_file(edge_path) if edge_path.is_file() else None
            ),
            "runtime_validation": edge_contract_runtime,
        },
        "interface_probe": interface_probe,
        "operator_results": results,
        "resolved_contract": resolved_contract,
        "security_boundary": {
            "p2_train_used": True if train_dataset is not None else False,
            "p2_validation_used": (
                True if validation_dataset is not None else False
            ),
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "full_training_performed": False,
            "smoke_optimizer_steps_completed": len(results),
            "scientific_checkpoint_created": False,
            "threshold_tuning_performed": False,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
            "legal_decoder_implemented": False,
        },
        "failures": failures,
        "warnings": warnings,
        "next_stage": (
            "V5_P2_G1C_SOURCE_ONLY_GRAPH_TRAINING_MATRIX"
            if not failures
            else "V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_FIX"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    contract_path = output_dir / "G1_SOURCE_ONLY_GRAPH_SMOKE_CONTRACT.json"
    if resolved_contract:
        write_json(contract_path, resolved_contract)

    markdown = [
        "# G1 Source-Only Graph Smoke Check",
        "",
        f"- Status: **{report['status']}**",
        f"- Device: `{report['device']}`",
        "- Historical failed smoke outputs touched: `false`",
        "",
        "This consolidated stage performs one optimizer step for Conv1D, "
        "GCNConv, GraphConv, and GATConv. It does not launch the twelve-run "
        "training matrix or select an architecture.",
        "",
    ]
    atomic_write(
        output_dir / "G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK.md",
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE if not failures else f"{STAGE}_HOLD",
        "report_sha256": sha256_file(report_path),
        "contract_file_sha256": (
            sha256_file(contract_path) if contract_path.is_file() else None
        ),
        "contract_sha256": resolved_contract.get("contract_sha256")
        if resolved_contract
        else None,
        "script_sha256": sha256_file(Path(__file__)),
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "topology_contract_sha256": canonical_topology.get(
            "contract_sha256"
        ),
        "edge_index_sha256": sha256_file(edge_path)
        if edge_path.is_file()
        else None,
        "historical_smoke_outputs_touched": False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "checkpoint_loaded": False,
        "b4_validation_cache_accessed": False,
        "b6_test_cache_accessed": False,
        "full_training_performed": False,
        "scientific_checkpoint_created": False,
        "architecture_selected": False,
        "quantization_performed": False,
        "rtl_generated": False,
        "legal_decoder_implemented": False,
        "next_stage": report["next_stage"],
    }
    write_json(output_dir / "G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_LOCK.json", lock)

    marker = COMPLETE if not failures else f"{STAGE}_HOLD"
    atomic_write(output_dir / marker, marker + "\n")

    print("===== G1 SOURCE-ONLY GRAPH SMOKE CHECK =====")
    print("status:", report["status"])
    print("historical_smoke_outputs_touched: false")
    print("failure_count:", len(failures))
    print("warning_count:", len(warnings))
    for failure in failures:
        print("FAIL:", failure)

    if not failures:
        print("device:", device)
        print("dataset_class:", dataset_class.__name__)
        print("train_batch_shape:", train_batch_contract["x_shape"])
        print(
            "validation_batch_shape:",
            validation_batch_contract["x_shape"],
        )
        print(
            "source_embedding_cutpoint:",
            interface_probe["cutpoint_name"],
        )
        print("source_embedding_dim:", interface_probe["embedding_dim"])
        print(
            "source_head_clone_max_abs_error:",
            interface_probe["clone_equivalence_max_abs_error"],
        )
        for operator in (
            "conv1d",
            "gcnconv",
            "graphconv",
            "gatconv",
        ):
            result = results[operator]
            print(
                f"{operator}: PASS "
                f"train_loss={result['train_loss_before_step']:.8f} "
                f"val_loss={result['validation_loss_after_step']:.8f} "
                f"params={result['total_parameter_count']}"
            )
        print(
            "contract_sha256:",
            resolved_contract["contract_sha256"],
        )

    print("p2_test_directory_enumerated: false")
    print("p2_test_tensors_deserialized: false")
    print("checkpoint_loaded: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("full_training_performed: false")
    print("architecture_selected: false")
    print("next_stage:", report["next_stage"])
    print(marker)
    return 0 if not failures else 1


def emergency_failure_marker(
    output_dir: Path,
    exc: BaseException,
) -> None:
    import traceback

    traceback.print_exc()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "stage": STAGE,
            "status": "HOLD",
            "decision": "UNEXPECTED_EXCEPTION_CAPTURED",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "historical_smoke_outputs_touched": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "full_training_performed": False,
            "architecture_selected": False,
            "next_stage": "V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_FIX",
        }
        write_json(output_dir / "UNEXPECTED_EXCEPTION.json", diagnostic)
        marker = f"{STAGE}_HOLD"
        atomic_write(output_dir / marker, marker + "\n")
        print(marker)
    except Exception:
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        print(f"STOP: output already exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)

    try:
        return run_smoke(args, output_dir)
    except BaseException as exc:
        emergency_failure_marker(output_dir, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
