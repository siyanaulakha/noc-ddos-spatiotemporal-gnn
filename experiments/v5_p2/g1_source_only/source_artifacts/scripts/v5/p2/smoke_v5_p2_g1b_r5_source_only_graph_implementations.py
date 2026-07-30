#!/usr/bin/env python3
"""
V5 P2-G1B-R5 Source-Only Graph Baseline Implementation Smoke Test

This append-only revision preserves the historical G1B, G1B-R1, G1B-R2, and
G1B-R4 HOLD outputs.

R4 loaded the topology report wrapper correctly but then looked for canonical
topology fields at the wrapper's top level. The actual frozen schema stores
those fields under `canonical_contract`. R5 explicitly unwraps and validates
that nested object before any model work.

The remaining R4 safeguards are preserved:

- flatten pair-aligned DataLoader batches before model input;
- require both attack and control rows in train/validation smoke batches;
- clone the exact B3 source-head architecture instead of inventing a new
  one-layer head;
- verify gradients reach the temporal backbone, source head, and each graph
  layer;
- verify identical B3/source-head initialization across all operators;
- report actual optimizer-step counts on partial failure;
- reset CUDA peak-memory accounting per operator;
- prohibit the full B3 wrapper from becoming the production G1C model.

The stage then resolves the exact B3 temporal-embedding cut point and performs one
finite forward/backward optimizer step for each G1 architecture:

- Conv1D-only
- Conv1D + GCNConv
- Conv1D + GraphConv
- Conv1D + GATConv

It reuses:
- pair manifests resolved in G1A-R1;
- the canonical static topology frozen in G1A-R2A;
- the existing untrained B3 model source as the common temporal backbone.

This is NOT full training. It creates no scientific checkpoint and selects no
architecture. The purpose is to prove that all four implementations share the
same temporal embeddings, tensor interface, labels, and graph batching before
the 12-run matrix is launched.

Forbidden:
- P2 test enumeration or tensor access;
- B4/B6 cache access;
- loading the frozen B3 checkpoint;
- threshold tuning;
- architecture selection;
- quantization, RTL generation, or Legal NoC decoder implementation.
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
from torch_geometric.nn import GATConv, GCNConv, GraphConv


STAGE = "V5_P2_G1B_R5_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE"
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
    max_scan: int = 256,
) -> tuple[list[int], dict[str, int]]:
    selected: list[int] = []
    attack_seen = False
    control_seen = False
    scanned = 0

    scan_limit = min(len(dataset), max_scan)
    for index in range(scan_limit):
        item = dataset[index]
        if not isinstance(item, dict) or "y_source" not in item:
            raise RuntimeError(
                f"dataset item {index} has no dictionary y_source target"
            )

        profile = source_row_profile(item["y_source"])
        attack_seen = attack_seen or profile["attack_rows"] > 0
        control_seen = control_seen or profile["control_rows"] > 0
        selected.append(index)
        scanned += 1

        if (
            len(selected) >= requested_items
            and attack_seen
            and control_seen
        ):
            break

    if not attack_seen or not control_seen:
        raise RuntimeError(
            "could not construct a smoke subset containing both attack "
            f"and control rows after scanning {scanned} items"
        )

    return selected, {
        "items_scanned": scanned,
        "items_selected": len(selected),
        "attack_seen": int(attack_seen),
        "control_seen": int(control_seen),
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

    attempts = []
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

    raise RuntimeError(f"unable to instantiate B3: {attempts}")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--g1b-hold-dir", type=Path, required=True)
    parser.add_argument("--g1b-r1-hold-dir", type=Path, required=True)
    parser.add_argument("--g1b-r2-hold-dir", type=Path, required=True)
    parser.add_argument("--g1b-r4-hold-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    g0_dir = args.g0_dir.expanduser().resolve()
    r1_dir = args.r1_dir.expanduser().resolve()
    topology_dir = args.topology_dir.expanduser().resolve()
    g1b_hold_dir = args.g1b_hold_dir.expanduser().resolve()
    g1b_r1_hold_dir = args.g1b_r1_hold_dir.expanduser().resolve()
    g1b_r2_hold_dir = args.g1b_r2_hold_dir.expanduser().resolve()
    g1b_r4_hold_dir = args.g1b_r4_hold_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    loader_path = args.loader.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        print(
            f"STOP: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    warnings: list[str] = []

    g0_lock_path = (
        g0_dir
        / "V5_P2_G0_GRAPH_BASELINE_AND_RTL_HANDOFF_PROTOCOL_LOCK.json"
    )
    r1_report_path = (
        r1_dir
        / "V5_P2_G1A_R1_SOURCE_ONLY_GRAPH_BASELINE_PREFLIGHT.json"
    )
    topology_contract_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json"
    )
    topology_lock_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json"
    )
    edge_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy"
    )
    topology_complete_path = (
        topology_dir
        / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_COMPLETE"
    )
    g1b_hold_report_path = (
        g1b_hold_dir
        / "V5_P2_G1B_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE.json"
    )
    g1b_hold_lock_path = (
        g1b_hold_dir
        / "V5_P2_G1B_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_LOCK.json"
    )
    g1b_hold_marker_path = (
        g1b_hold_dir
        / "V5_P2_G1B_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
    )
    g1b_r1_hold_report_path = (
        g1b_r1_hold_dir
        / "V5_P2_G1B_R1_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE.json"
    )
    g1b_r1_hold_lock_path = (
        g1b_r1_hold_dir
        / "V5_P2_G1B_R1_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_LOCK.json"
    )
    g1b_r1_hold_marker_path = (
        g1b_r1_hold_dir
        / "V5_P2_G1B_R1_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
    )
    g1b_r2_hold_report_path = (
        g1b_r2_hold_dir
        / "V5_P2_G1B_R2_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE.json"
    )
    g1b_r2_hold_lock_path = (
        g1b_r2_hold_dir
        / "V5_P2_G1B_R2_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_LOCK.json"
    )
    g1b_r2_hold_marker_path = (
        g1b_r2_hold_dir
        / "V5_P2_G1B_R2_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
    )
    g1b_r4_hold_report_path = (
        g1b_r4_hold_dir
        / "V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE.json"
    )
    g1b_r4_hold_lock_path = (
        g1b_r4_hold_dir
        / "V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_LOCK.json"
    )
    g1b_r4_hold_marker_path = (
        g1b_r4_hold_dir
        / "V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
    )

    prerequisites = (
        g0_lock_path,
        r1_report_path,
        topology_contract_path,
        topology_lock_path,
        edge_path,
        topology_complete_path,
        g1b_hold_report_path,
        g1b_hold_lock_path,
        g1b_hold_marker_path,
        g1b_r1_hold_report_path,
        g1b_r1_hold_lock_path,
        g1b_r1_hold_marker_path,
        g1b_r2_hold_report_path,
        g1b_r2_hold_lock_path,
        g1b_r2_hold_marker_path,
        g1b_r4_hold_report_path,
        g1b_r4_hold_lock_path,
        g1b_r4_hold_marker_path,
        loader_path,
        model_path,
    )
    for path in prerequisites:
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if not data_root.is_dir():
        failures.append(f"missing data root: {data_root}")

    if not failures:
        g0_lock = json.loads(
            g0_lock_path.read_text(encoding="utf-8")
        )
        r1_report = json.loads(
            r1_report_path.read_text(encoding="utf-8")
        )
        topology_report = json.loads(
            topology_contract_path.read_text(encoding="utf-8")
        )
        topology_lock = json.loads(
            topology_lock_path.read_text(encoding="utf-8")
        )

        g1b_hold_report = json.loads(
            g1b_hold_report_path.read_text(encoding="utf-8")
        )
        g1b_hold_lock = json.loads(
            g1b_hold_lock_path.read_text(encoding="utf-8")
        )
        g1b_r1_hold_report = json.loads(
            g1b_r1_hold_report_path.read_text(encoding="utf-8")
        )
        g1b_r1_hold_lock = json.loads(
            g1b_r1_hold_lock_path.read_text(encoding="utf-8")
        )
        g1b_r2_hold_report = json.loads(
            g1b_r2_hold_report_path.read_text(encoding="utf-8")
        )
        g1b_r2_hold_lock = json.loads(
            g1b_r2_hold_lock_path.read_text(encoding="utf-8")
        )
        g1b_r4_hold_report = json.loads(
            g1b_r4_hold_report_path.read_text(encoding="utf-8")
        )
        g1b_r4_hold_lock = json.loads(
            g1b_r4_hold_lock_path.read_text(encoding="utf-8")
        )

        if g0_lock.get("protocol_sha256") != EXPECTED_G0_PROTOCOL_SHA:
            failures.append("G0 protocol SHA changed")

        # The topology JSON is a stage report wrapper. The immutable topology
        # itself is nested under `canonical_contract`.
        if topology_report.get("status") != "COMPLETE":
            failures.append("topology stage report is not COMPLETE")
        if (
            topology_report.get("decision")
            != "FREEZE_CANONICAL_ROW_MAJOR_4X4_TOPOLOGY_AND_AUTHORIZE_G1"
        ):
            failures.append("topology stage decision changed")

        canonical_topology = topology_report.get("canonical_contract")
        if not isinstance(canonical_topology, dict):
            failures.append(
                "topology report has no dictionary canonical_contract"
            )
            canonical_topology = {}

        actual_contract_file_sha = sha256_file(topology_contract_path)
        actual_edge_file_sha = sha256_file(edge_path)

        if (
            topology_lock.get("contract_file_sha256")
            != actual_contract_file_sha
        ):
            failures.append("topology report file SHA disagrees with lock")
        if (
            topology_lock.get("report_sha256")
            != actual_contract_file_sha
        ):
            failures.append("topology report SHA disagrees with lock")
        if (
            topology_lock.get("contract_sha256")
            != canonical_topology.get("contract_sha256")
        ):
            failures.append(
                "nested canonical topology SHA disagrees with lock"
            )
        if topology_lock.get("edge_index_sha256") != actual_edge_file_sha:
            failures.append("canonical edge artifact SHA mismatch")
        if actual_edge_file_sha != EXPECTED_EDGE_ARTIFACT_SHA:
            failures.append("canonical edge artifact changed")

        contract_edge_artifact = (
            canonical_topology
            .get("artifacts", {})
            .get("edge_index", {})
        )
        contract_shape = contract_edge_artifact.get("shape")
        contract_edge_sha = contract_edge_artifact.get("sha256")
        physical_graph = canonical_topology.get(
            "physical_routing_graph", {}
        )

        if contract_edge_sha != actual_edge_file_sha:
            failures.append(
                "nested topology contract edge SHA disagrees with artifact"
            )
        if canonical_topology.get("topology") != "4x4_2D_MESH":
            failures.append("topology contract is not a 4x4 2D mesh")
        if canonical_topology.get("num_nodes") != 16:
            failures.append("topology contract does not contain 16 nodes")
        if canonical_topology.get("router_numbering") != "row_major":
            failures.append("topology router numbering is not row-major")
        if contract_shape != [2, 48]:
            failures.append(
                f"topology contract edge shape is {contract_shape}, "
                "expected [2,48]"
            )
        if physical_graph.get("directed_edge_count") != 48:
            failures.append(
                "topology contract does not freeze 48 directed edges"
            )
        if physical_graph.get("undirected_link_count") != 24:
            failures.append(
                "topology contract does not freeze 24 physical links"
            )
        if physical_graph.get("physical_self_loops") is not False:
            failures.append(
                "topology contract unexpectedly permits physical self-loops"
            )

        if (
            g1b_hold_marker_path.read_text(encoding="utf-8").strip()
            != "V5_P2_G1B_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
        ):
            failures.append("historical G1B HOLD marker changed")
        if g1b_hold_lock.get("report_sha256") != sha256_file(
            g1b_hold_report_path
        ):
            failures.append("historical G1B HOLD report SHA mismatch")
        if g1b_hold_report.get("status") != "HOLD":
            failures.append("historical G1B report is not HOLD")

        if (
            g1b_r1_hold_marker_path.read_text(encoding="utf-8").strip()
            != "V5_P2_G1B_R1_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
        ):
            failures.append("historical G1B-R1 HOLD marker changed")
        if g1b_r1_hold_lock.get("report_sha256") != sha256_file(
            g1b_r1_hold_report_path
        ):
            failures.append("historical G1B-R1 HOLD report SHA mismatch")
        if g1b_r1_hold_report.get("status") != "HOLD":
            failures.append("historical G1B-R1 report is not HOLD")

        if (
            g1b_r2_hold_marker_path.read_text(encoding="utf-8").strip()
            != "V5_P2_G1B_R2_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
        ):
            failures.append("historical G1B-R2 HOLD marker changed")
        if g1b_r2_hold_lock.get("report_sha256") != sha256_file(
            g1b_r2_hold_report_path
        ):
            failures.append("historical G1B-R2 HOLD report SHA mismatch")
        if g1b_r2_hold_report.get("status") != "HOLD":
            failures.append("historical G1B-R2 report is not HOLD")

        if (
            g1b_r4_hold_marker_path.read_text(encoding="utf-8").strip()
            != "V5_P2_G1B_R4_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_HOLD"
        ):
            failures.append("historical G1B-R4 HOLD marker changed")
        if g1b_r4_hold_lock.get("report_sha256") != sha256_file(
            g1b_r4_hold_report_path
        ):
            failures.append("historical G1B-R4 HOLD report SHA mismatch")
        if g1b_r4_hold_report.get("status") != "HOLD":
            failures.append("historical G1B-R4 report is not HOLD")

        if r1_report.get("status") != "HOLD":
            warnings.append(
                "G1A-R1 report is not HOLD; using its resolved manifest "
                "provenance regardless"
            )

    results: dict[str, Any] = {}
    resolved_contract: dict[str, Any] = {}

    if not failures:
        selected = (
            r1_report
            .get("manifest_discovery", {})
            .get("selected", {})
        )
        train_manifest_text = selected.get("train_manifest")
        validation_manifest_text = selected.get(
            "validation_manifest"
        )
        dataset_class_name = (
            r1_report.get("loader", {}).get("selected_class")
        )

        if not train_manifest_text:
            failures.append("R1 report has no train manifest")
        if not validation_manifest_text:
            failures.append("R1 report has no validation manifest")

    if not failures:
        train_manifest = Path(train_manifest_text).resolve()
        validation_manifest = Path(
            validation_manifest_text
        ).resolve()

        loader_module = import_module_from_path(
            "v5_p2_g1b_loader",
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

        requested_smoke_items = max(args.batch_size, 8)
        train_indices, train_selection = (
            select_balanced_smoke_indices(
                train_dataset,
                requested_items=requested_smoke_items,
            )
        )
        validation_indices, validation_selection = (
            select_balanced_smoke_indices(
                validation_dataset,
                requested_items=requested_smoke_items,
            )
        )

        smoke_train_count = len(train_indices)
        smoke_validation_count = len(validation_indices)

        train_subset = Subset(
            train_dataset,
            train_indices,
        )
        validation_subset = Subset(
            validation_dataset,
            validation_indices,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        validation_loader = DataLoader(
            validation_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        train_batch_raw = next(iter(train_loader))
        validation_batch_raw = next(
            iter(validation_loader)
        )
        train_batch_cpu, train_batch_contract = (
            flatten_pair_aligned_batch(train_batch_raw)
        )
        validation_batch_cpu, validation_batch_contract = (
            flatten_pair_aligned_batch(validation_batch_raw)
        )

        if "x" not in train_batch_cpu:
            failures.append("train batch has no x key")
        if "y_source" not in train_batch_cpu:
            failures.append("train batch has no y_source key")

    if not failures:
        base_edge_index = torch.from_numpy(
            np.load(edge_path, allow_pickle=False)
        ).to(dtype=torch.long)

        if list(base_edge_index.shape) != [2, 48]:
            failures.append(
                "canonical edge_index does not have shape [2,48]"
            )

    if not failures:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_module = import_module_from_path(
            "v5_p2_g1b_model",
            model_path,
        )

        try:
            embedding_contract = infer_embedding_dim(
                model_module,
                train_batch_cpu,
                device,
            )
            embedding_dim = embedding_contract[
                "embedding_dim"
            ]

            for operator in (
                "conv1d",
                "gcnconv",
                "graphconv",
                "gatconv",
            ):
                results[operator] = smoke_one_operator(
                    operator=operator,
                    model_module=model_module,
                    train_batch_cpu=train_batch_cpu,
                    validation_batch_cpu=validation_batch_cpu,
                    base_edge_index=base_edge_index,
                    device=device,
                    embedding_dim=embedding_dim,
                    seed=args.seed,
                )

            cutpoints = {
                value["source_cutpoint_name"]
                for value in results.values()
            }
            dimensions = {
                value["embedding_dim"]
                for value in results.values()
            }
            forward_variants = {
                value["b3_forward_variant"]
                for value in results.values()
            }
            initial_b3_hashes = {
                value["initial_b3_state_sha256"]
                for value in results.values()
            }
            initial_source_head_hashes = {
                value["initial_source_head_state_sha256"]
                for value in results.values()
            }

            if len(cutpoints) != 1:
                failures.append(
                    f"operators used different cutpoints: {cutpoints}"
                )
            if len(dimensions) != 1:
                failures.append(
                    f"operators used different embedding dims: {dimensions}"
                )
            if len(forward_variants) != 1:
                failures.append(
                    "operators used different B3 call variants: "
                    f"{forward_variants}"
                )
            if len(initial_b3_hashes) != 1:
                failures.append(
                    "operators did not start from identical B3 weights"
                )
            if len(initial_source_head_hashes) != 1:
                failures.append(
                    "operators did not start from identical source-head "
                    "weights"
                )

            resolved_contract_core = {
                "name": (
                    "V5_P2_G1B_SOURCE_ONLY_GRAPH_IMPLEMENTATION_CONTRACT"
                ),
                "version": 1,
                "dataset_class": dataset_class.__name__,
                "train_manifest": str(train_manifest),
                "validation_manifest": str(
                    validation_manifest
                ),
                "input_key": "x",
                "source_target_key": "y_source",
                "source_target_type": (
                    "16 independent binary source labels"
                ),
                "train_batch_contract": train_batch_contract,
                "validation_batch_contract": (
                    validation_batch_contract
                ),
                "train_subset_selection": train_selection,
                "validation_subset_selection": (
                    validation_selection
                ),
                "common_temporal_backbone": (
                    embedding_contract["b3_class"]
                ),
                "common_temporal_backbone_parameter_count": (
                    embedding_contract["b3_parameter_count"]
                ),
                "source_embedding_cutpoint": (
                    embedding_contract["cutpoint_name"]
                ),
                "source_embedding_cutpoint_class": (
                    embedding_contract["cutpoint_class"]
                ),
                "source_head_policy": (
                    "clone the exact B3 source-head architecture and "
                    "initial state for every operator"
                ),
                "common_initial_b3_state_sha256": next(
                    iter(initial_b3_hashes)
                ),
                "common_initial_source_head_state_sha256": next(
                    iter(initial_source_head_hashes)
                ),
                "b3_forward_variant": (
                    embedding_contract["forward_variant"]
                ),
                "node_embedding_shape_smoke": (
                    embedding_contract["embedding_shape"]
                ),
                "node_embedding_dim": embedding_dim,
                "num_nodes": 16,
                "physical_edge_index": {
                    "path": str(edge_path),
                    "sha256": sha256_file(edge_path),
                    "shape": [2, 48],
                    "batching": (
                        "offset node indices by 16 per graph; "
                        "cross-graph edges forbidden"
                    ),
                },
                "graph_layers": 2,
                "graph_hidden_width": embedding_dim,
                "common_activation_for_smoke": "ReLU",
                "operators": [
                    "conv1d",
                    "gcnconv",
                    "graphconv",
                    "gatconv",
                ],
                "initial_training_seeds": [107, 117, 127],
                "execution_policy": (
                    "serial one-model-at-a-time queue"
                ),
                "scientific_training_authorized": False,
                "production_model_requirements": {
                    "full_b3_wrapper_allowed": False,
                    "unused_graph_count_transit_victim_path_heads_allowed": (
                        False
                    ),
                    "clean_temporal_encoder_extraction_required": True,
                    "clean_encoder_embedding_equivalence_max_abs_error": (
                        1.0e-6
                    ),
                    "reason": (
                        "the hook-based full-B3 adapter is only an "
                        "interface oracle for smoke testing; G1C must not "
                        "carry unused B3 task heads into source-only "
                        "parameter, FLOP, memory, or RTL comparisons"
                    ),
                },
                "topology_provenance_validation": (
                    "contract-file SHA, canonical contract SHA, and edge "
                    "artifact SHA are verified relationally against the "
                    "frozen topology lock"
                ),
                "next_required_stage": (
                    "V5_P2_G1C_SOURCE_ONLY_GRAPH_TRAINING_MATRIX"
                ),
            }
            resolved_contract = {
                **resolved_contract_core,
                "contract_sha256": canonical_sha256(
                    resolved_contract_core
                ),
            }
        except Exception as exc:
            failures.append(
                "implementation smoke failed: "
                f"{type(exc).__name__}: {exc}"
            )

    report = {
        "stage": STAGE,
        "status": "COMPLETE" if not failures else "HOLD",
        "historical_g1b_hold_preserved": True,
        "historical_g1b_r1_hold_preserved": True,
        "historical_g1b_r2_hold_preserved": True,
        "historical_g1b_r4_hold_preserved": True,
        "topology_schema": {
            "wrapper_key": "canonical_contract",
            "wrapper_status": (
                topology_report.get("status")
                if "topology_report" in locals()
                else None
            ),
            "canonical_contract_sha256": (
                canonical_topology.get("contract_sha256")
                if "canonical_topology" in locals()
                else None
            ),
        },
        "decision": (
            "FREEZE_SHARED_B3_NODE_EMBEDDING_INTERFACE_AND_AUTHORIZE_G1C"
            if not failures
            else "HOLD_G1C_PENDING_IMPLEMENTATION_FIX"
        ),
        "device": (
            str(device)
            if "device" in locals()
            else None
        ),
        "torch_version": torch.__version__,
        "torch_geometric_operators": [
            "GCNConv",
            "GraphConv",
            "GATConv",
        ],
        "train_subset_items": (
            smoke_train_count
            if "smoke_train_count" in locals()
            else None
        ),
        "validation_subset_items": (
            smoke_validation_count
            if "smoke_validation_count" in locals()
            else None
        ),
        "batch_size": args.batch_size,
        "smoke_seed": args.seed,
        "train_subset_selection": (
            train_selection
            if "train_selection" in locals()
            else None
        ),
        "validation_subset_selection": (
            validation_selection
            if "validation_selection" in locals()
            else None
        ),
        "train_batch_contract": (
            train_batch_contract
            if "train_batch_contract" in locals()
            else None
        ),
        "validation_batch_contract": (
            validation_batch_contract
            if "validation_batch_contract" in locals()
            else None
        ),
        "resolved_implementation_contract": (
            resolved_contract
        ),
        "operator_results": results,
        "security_boundary": {
            "train_items_available": (
                len(train_dataset)
                if "train_dataset" in locals()
                else None
            ),
            "validation_items_available": (
                len(validation_dataset)
                if "validation_dataset" in locals()
                else None
            ),
            "test_directory_enumerated": False,
            "test_tensors_deserialized": False,
            "checkpoint_loaded": False,
            "b4_validation_cache_accessed": False,
            "b6_test_cache_accessed": False,
            "full_training_performed": False,
            "smoke_optimizer_steps_completed": len(results),
            "smoke_optimizer_steps_planned": 4,
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
            else "V5_P2_G1B_R6_IMPLEMENTATION_FIX"
        ),
    }

    report_path = output_dir / f"{STAGE}.json"
    write_json(report_path, report)

    contract_path = (
        output_dir
        / "V5_P2_G1B_R5_SOURCE_ONLY_GRAPH_IMPLEMENTATION_CONTRACT.json"
    )
    if resolved_contract:
        write_json(contract_path, resolved_contract)

    markdown = [
        "# V5 P2-G1B-R5 Source-Only Graph Implementation Smoke",
        "",
        f"- Status: **{report['status']}**",
        f"- Device: `{report['device']}`",
        f"- Batch size: `{args.batch_size}`",
        "",
        "This stage performs exactly one smoke optimizer step per "
        "architecture. It does not conduct the scientific three-seed "
        "training matrix or select an architecture.",
        "",
        "Architectures:",
        "",
        "- Conv1D-only",
        "- Conv1D + GCNConv",
        "- Conv1D + GraphConv",
        "- Conv1D + GATConv",
        "",
        "The next stage may launch the serial 12-run matrix only after "
        "building a clean source-only model that extracts the temporal "
        "encoder and removes all unused B3 task heads. The hook-based full "
        "B3 wrapper in this smoke is an interface oracle, not the production "
        "training architecture.",
        "",
    ]
    atomic_write(
        output_dir
        / "V5_P2_G1B_R5_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE.md",
        "\n".join(markdown),
    )

    lock = {
        "status": COMPLETE if not failures else f"{STAGE}_HOLD",
        "report_sha256": sha256_file(report_path),
        "implementation_contract_file_sha256": (
            sha256_file(contract_path)
            if contract_path.is_file()
            else None
        ),
        "implementation_contract_sha256": (
            resolved_contract.get("contract_sha256")
            if resolved_contract
            else None
        ),
        "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA,
        "topology_contract_sha256": (
            canonical_topology.get("contract_sha256")
            if "canonical_topology" in locals()
            else None
        ),
        "topology_wrapper_key": "canonical_contract",
        "topology_contract_file_sha256": (
            sha256_file(topology_contract_path)
            if topology_contract_path.is_file()
            else None
        ),
        "historical_g1b_hold_report_sha256": (
            sha256_file(g1b_hold_report_path)
            if g1b_hold_report_path.is_file()
            else None
        ),
        "historical_g1b_r1_hold_report_sha256": (
            sha256_file(g1b_r1_hold_report_path)
            if g1b_r1_hold_report_path.is_file()
            else None
        ),
        "historical_g1b_r2_hold_report_sha256": (
            sha256_file(g1b_r2_hold_report_path)
            if g1b_r2_hold_report_path.is_file()
            else None
        ),
        "historical_g1b_r4_hold_report_sha256": (
            sha256_file(g1b_r4_hold_report_path)
            if g1b_r4_hold_report_path.is_file()
            else None
        ),
        "historical_g1b_hold_preserved": True,
        "historical_g1b_r1_hold_preserved": True,
        "historical_g1b_r2_hold_preserved": True,
        "historical_g1b_r4_hold_preserved": True,
        "edge_index_sha256": sha256_file(edge_path)
        if edge_path.is_file()
        else None,
        "test_directory_enumerated": False,
        "test_tensors_deserialized": False,
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
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(
        output_dir
        / "V5_P2_G1B_R5_SOURCE_ONLY_GRAPH_IMPLEMENTATION_SMOKE_LOCK.json",
        lock,
    )

    if failures:
        atomic_write(
            output_dir / f"{STAGE}_HOLD",
            f"{STAGE}_HOLD\n",
        )
        print("===== V5 P2-G1B-R5 SOURCE-ONLY GRAPH SMOKE =====")
        print("status: HOLD")
        print("failure_count:", len(failures))
        print("warning_count:", len(warnings))
        for failure in failures:
            print("FAIL:", failure)
        print("next_stage: V5_P2_G1B_R6_IMPLEMENTATION_FIX")
        print(f"{STAGE}_HOLD")
        return 1

    atomic_write(
        output_dir / COMPLETE,
        COMPLETE + "\n",
    )

    print("===== V5 P2-G1B-R5 SOURCE-ONLY GRAPH SMOKE =====")
    print("status: COMPLETE")
    print("historical_g1b_hold_preserved: true")
    print("historical_g1b_r1_hold_preserved: true")
    print("historical_g1b_r2_hold_preserved: true")
    print("historical_g1b_r4_hold_preserved: true")
    print("topology_wrapper_key: canonical_contract")
    print(
        "topology_contract_sha256:",
        canonical_topology["contract_sha256"],
    )
    print(
        "decision: "
        "FREEZE_SHARED_B3_NODE_EMBEDDING_INTERFACE_AND_AUTHORIZE_G1C"
    )
    print("device:", device)
    print("dataset_class:", dataset_class.__name__)
    print("train_items:", len(train_dataset))
    print("validation_items:", len(validation_dataset))
    print("batch_size:", args.batch_size)
    print(
        "train_pair_aligned_input:",
        train_batch_contract["pair_aligned_input"],
    )
    print(
        "train_flattened_x_shape:",
        train_batch_contract["flattened_x_shape"],
    )
    print(
        "train_attack_rows:",
        train_batch_contract["attack_rows"],
    )
    print(
        "train_control_rows:",
        train_batch_contract["control_rows"],
    )
    print(
        "source_embedding_cutpoint:",
        resolved_contract["source_embedding_cutpoint"],
    )
    print(
        "node_embedding_dim:",
        resolved_contract["node_embedding_dim"],
    )
    print(
        "b3_forward_variant:",
        resolved_contract["b3_forward_variant"],
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
            f"params={result['total_parameter_count']} "
            f"grad_tensors="
            f"{result['gradient_summary']['parameter_tensors_with_gradient']}"
        )

    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    print("checkpoint_loaded: false")
    print("b4_validation_cache_accessed: false")
    print("b6_test_cache_accessed: false")
    print("full_training_performed: false")
    print("smoke_optimizer_steps_completed:", len(results))
    print("scientific_checkpoint_created: false")
    print("architecture_selected: false")
    print("failure_count: 0")
    print("warning_count:", len(warnings))
    print(
        "implementation_contract_sha256:",
        resolved_contract["contract_sha256"],
    )
    print(
        "next_stage: "
        "V5_P2_G1C_SOURCE_ONLY_GRAPH_TRAINING_MATRIX"
    )
    print(COMPLETE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
