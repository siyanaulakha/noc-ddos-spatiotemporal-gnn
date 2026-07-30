#!/usr/bin/env python3
"""
V5 P2-G1C source-only graph training: one architecture/seed run.

This stage trains exactly one source-only model from:
    conv1d, gcnconv, graphconv, gatconv
with exactly one initial-screening seed from:
    107, 117, 127

Authorized data are P2 train and validation only. The script never lists,
opens, constructs, or evaluates the P2 test split. Threshold selection is
performed only after the best validation checkpoint has been selected.

The production model is clean: it copies only the frozen B3 temporal encoder,
node projection, and source head. It does not retain unused graph/count/transit/
victim/path heads and does not use forward hooks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from torch_geometric.nn import GATConv, GCNConv, GraphConv
except Exception as exc:  # pragma: no cover - converted to controlled HOLD
    GATConv = None
    GCNConv = None
    GraphConv = None
    PYG_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    PYG_IMPORT_ERROR = None


STAGE = "V5_P2_G1C_SOURCE_ONLY_GRAPH_SINGLE_RUN"
COMPLETE = f"{STAGE}_COMPLETE"
HOLD = f"{STAGE}_HOLD"

OPERATORS = ("conv1d", "gcnconv", "graphconv", "gatconv")
SEEDS = (107, 117, 127)
EXPECTED_TRAIN_ITEMS = 70_166
EXPECTED_VALIDATION_ITEMS = 12_528
EXPECTED_LOADER_SHA = (
    "2725ff993f4f03ebee3d5ffb775b1fedd6b131a9c4c89ed8049f45b249c24ac2"
)
EXPECTED_B3_MODEL_SHA = (
    "56ee3207d039b60e8e3a898a689cd8e361247c86b7450a3a5e95f423ebe30def"
)
EXPECTED_EDGE_SHA = (
    "f6b8050bc158de509b0ff1c5d1d7cb1ffe32c08b0f2287398270b1b891b57aff"
)
EXPECTED_TOPOLOGY_CONTRACT_SHA = (
    "74112ce145cd3347148d4c784be125add7945943e80e98948368f4c076d047af"
)
EXPECTED_SMOKE_CONTRACT_SHA = (
    "b8871e8516c05f54d37c5a761b3e17cbf91b0722f08e63b37a93251f6e30e5b6"
)
EXPECTED_SMOKE_STATUS = "V5_P2_G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_COMPLETE"
EXPECTED_FULL_B3_PARAMETERS = 43_273
EXPECTED_CLEAN_PARAMETERS = {
    "conv1d": 28_353,
    "gcnconv": 36_673,
    "graphconv": 44_865,
    "gatconv": 36_929,
}

MAX_EPOCHS = 100
MIN_EPOCHS = 15
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4
ITEM_BATCH_SIZE = 256
PAIR_BLOCK_BATCH_SIZE = ITEM_BATCH_SIZE // 2
GRADIENT_CLIP = 1.0
THRESHOLD_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)

CHECKPOINT_WEIGHTS = {
    "source_average_precision": 0.50,
    "source_auroc": 0.25,
    "derived_graph_average_precision": 0.25,
}
THRESHOLD_WEIGHTS = {
    "source_node_f1": 0.60,
    "source_exact_set_attack": 0.40,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> str:
    payload = OrderedDict(
        (key, value.detach().cpu().contiguous())
        for key, value in state_dict.items()
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_auroc(truth: np.ndarray, score: np.ndarray) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC undefined because one binary class is absent")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop

    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2
        )
        / (positive_count * negative_count)
    )


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    y = truth.astype(np.int64).reshape(-1)
    s = score.astype(np.float64).reshape(-1)
    positive_count = int((y == 1).sum())
    if positive_count == 0:
        raise ValueError("average precision undefined because positives are absent")
    order = np.argsort(-s, kind="mergesort")
    sorted_truth = y[order]
    cumulative_positive = np.cumsum(sorted_truth)
    precision_at_rank = cumulative_positive / np.arange(1, len(y) + 1)
    return float((precision_at_rank * sorted_truth).sum() / positive_count)


def binary_metrics(
    truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = truth.astype(np.int64).reshape(-1)
    prediction = (score.reshape(-1) >= threshold).astype(np.int64)
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    tnr = safe_divide(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": safe_divide(fp, fp + tn),
        "tnr": tnr,
        "balanced_accuracy": 0.5 * (recall + tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def exact_set_metrics(
    truth: np.ndarray,
    source_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = truth.astype(np.int64)
    prediction = (source_score >= threshold).astype(np.int64)
    exact = np.all(prediction == y, axis=1)
    attack = np.any(y == 1, axis=1)
    control = ~attack
    return {
        "overall": float(exact.mean()),
        "attack": float(exact[attack].mean()) if bool(attack.any()) else 0.0,
        "control": float(exact[control].mean()) if bool(control.any()) else 0.0,
    }


def source_and_graph_metrics(
    truth_2d: np.ndarray,
    score_2d: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if truth_2d.shape != score_2d.shape or truth_2d.ndim != 2:
        raise ValueError("source truth/score must be matching [items,16] arrays")
    source_truth = truth_2d.reshape(-1)
    source_score = score_2d.reshape(-1)
    graph_truth = np.any(truth_2d >= 1, axis=1).astype(np.int64)
    graph_score = np.max(score_2d, axis=1)
    return {
        "source": {
            "auroc": binary_auroc(source_truth, source_score),
            "average_precision": average_precision(source_truth, source_score),
            "thresholded": binary_metrics(source_truth, source_score, threshold),
            "positive_entries": int((source_truth == 1).sum()),
            "negative_entries": int((source_truth == 0).sum()),
        },
        "derived_graph": {
            "auroc": binary_auroc(graph_truth, graph_score),
            "average_precision": average_precision(graph_truth, graph_score),
            "thresholded": binary_metrics(graph_truth, graph_score, threshold),
            "positive_items": int((graph_truth == 1).sum()),
            "negative_items": int((graph_truth == 0).sum()),
        },
        "exact_set": exact_set_metrics(truth_2d, score_2d, threshold),
    }


def checkpoint_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        CHECKPOINT_WEIGHTS["source_average_precision"]
        * metrics["source"]["average_precision"]
        + CHECKPOINT_WEIGHTS["source_auroc"]
        * metrics["source"]["auroc"]
        + CHECKPOINT_WEIGHTS["derived_graph_average_precision"]
        * metrics["derived_graph"]["average_precision"]
    )


def tune_source_threshold(
    truth_2d: np.ndarray,
    score_2d: np.ndarray,
) -> dict[str, Any]:
    """Tune a fixed validation-only 0.001 grid without repeated ranking sorts."""
    if truth_2d.shape != score_2d.shape or truth_2d.ndim != 2:
        raise ValueError("threshold inputs must be matching [items,16] arrays")

    truth_bool = truth_2d.astype(bool, copy=False)
    score = score_2d.astype(np.float64, copy=False)
    graph_truth = truth_bool.any(axis=1)
    attack_selector = graph_truth
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None

    # Chunking bounds temporary prediction storage to roughly 6.5 MB for the
    # frozen 12,528 x 16 validation layout.
    chunk_size = 32
    for start in range(0, len(THRESHOLD_GRID), chunk_size):
        thresholds = THRESHOLD_GRID[start:start + chunk_size]
        prediction = score[None, :, :] >= thresholds[:, None, None]

        tp = np.logical_and(prediction, truth_bool[None, :, :]).sum(axis=(1, 2))
        fp = np.logical_and(prediction, ~truth_bool[None, :, :]).sum(axis=(1, 2))
        fn = np.logical_and(~prediction, truth_bool[None, :, :]).sum(axis=(1, 2))
        precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0
        )
        recall = np.divide(
            tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0
        )
        source_f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        exact = np.all(prediction == truth_bool[None, :, :], axis=2)
        exact_attack = exact[:, attack_selector].mean(axis=1)

        graph_prediction = prediction.any(axis=2)
        graph_tp = np.logical_and(
            graph_prediction, graph_truth[None, :]
        ).sum(axis=1)
        graph_tn = np.logical_and(
            ~graph_prediction, ~graph_truth[None, :]
        ).sum(axis=1)
        graph_fp = np.logical_and(
            graph_prediction, ~graph_truth[None, :]
        ).sum(axis=1)
        graph_fn = np.logical_and(
            ~graph_prediction, graph_truth[None, :]
        ).sum(axis=1)
        graph_tpr = np.divide(
            graph_tp,
            graph_tp + graph_fn,
            out=np.zeros_like(graph_tp, dtype=np.float64),
            where=(graph_tp + graph_fn) != 0,
        )
        graph_tnr = np.divide(
            graph_tn,
            graph_tn + graph_fp,
            out=np.zeros_like(graph_tn, dtype=np.float64),
            where=(graph_tn + graph_fp) != 0,
        )
        graph_balanced = 0.5 * (graph_tpr + graph_tnr)
        objective = (
            THRESHOLD_WEIGHTS["source_node_f1"] * source_f1
            + THRESHOLD_WEIGHTS["source_exact_set_attack"] * exact_attack
        )

        for index, threshold in enumerate(thresholds):
            row = {
                "threshold": float(threshold),
                "objective": float(objective[index]),
                "source_node_f1": float(source_f1[index]),
                "source_exact_set_attack": float(exact_attack[index]),
                "derived_graph_balanced_accuracy": float(graph_balanced[index]),
            }
            rows.append(row)
            rank = (
                row["objective"],
                row["source_node_f1"],
                row["source_exact_set_attack"],
                row["derived_graph_balanced_accuracy"],
                -abs(row["threshold"] - 0.5),
                -row["threshold"],
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_row = row

    if best_row is None or best_rank is None:
        raise RuntimeError("threshold selection produced no candidate")
    selected_metrics = source_and_graph_metrics(
        truth_2d, score_2d, best_row["threshold"]
    )
    selected = {
        **best_row,
        "metrics": selected_metrics,
        "ranking_tuple": list(best_rank),
    }
    return {
        "grid": {
            "kind": "uniform_closed_interval",
            "start": 0.0,
            "stop": 1.0,
            "step": 0.001,
            "candidate_count": len(THRESHOLD_GRID),
        },
        "objective_weights": THRESHOLD_WEIGHTS,
        "selected": selected,
        "rows": rows,
    }


class PairBlockBatchSampler(Sampler[list[int]]):
    """Deterministic pair-aligned ATTACK/CONTROL block sampler."""

    def __init__(
        self,
        dataset,
        *,
        block_batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.block_batch_size = int(block_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.block_batch_size <= 0:
            raise ValueError("block_batch_size must be positive")
        if len(dataset) % 2 != 0:
            raise ValueError("dataset item count must be even")

        groups: OrderedDict[str, list[int]] = OrderedDict()
        for base in range(0, len(dataset._index), 2):
            attack = dataset._index[base]
            control = dataset._index[base + 1]
            if (
                attack.mode != "attack"
                or control.mode != "control"
                or attack.pair_key != control.pair_key
                or attack.start != control.start
                or attack.target != control.target
            ):
                raise ValueError(f"pair-block contract failed at base {base}")
            groups.setdefault(attack.pair_key, []).append(base)
        self.groups = groups
        self.block_count = sum(len(group) for group in groups.values())
        if self.block_count * 2 != len(dataset):
            raise RuntimeError("pair-block count does not cover dataset")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.block_count / self.block_batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pair_keys = list(self.groups)
        if self.shuffle:
            rng.shuffle(pair_keys)
        ordered_blocks: list[int] = []
        for pair_key in pair_keys:
            blocks = list(self.groups[pair_key])
            if self.shuffle:
                rng.shuffle(blocks)
            ordered_blocks.extend(blocks)
        for start in range(0, len(ordered_blocks), self.block_batch_size):
            item_indices: list[int] = []
            for base in ordered_blocks[start:start + self.block_batch_size]:
                item_indices.extend((base, base + 1))
            yield item_indices


def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    base = base_edge_index.to(device=device, dtype=torch.long)
    result = torch.cat(
        [base + graph_index * num_nodes for graph_index in range(batch_size)],
        dim=1,
    )
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
        raise RuntimeError("batched edge_index contains cross-graph edges")
    return result


def validate_edge_index(edge_index: torch.Tensor) -> dict[str, Any]:
    edge = edge_index.detach().cpu().long().contiguous()
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"edge_index shape={tuple(edge.shape)}, expected (2,48)")
    if int(edge.min()) != 0 or int(edge.max()) != 15:
        raise ValueError("edge_index router range is not exactly 0..15")
    if int((edge[0] == edge[1]).sum()) != 0:
        raise ValueError("stored physical edge_index contains self-loops")
    edge_set = {tuple(map(int, edge_pair)) for edge_pair in edge.t().tolist()}
    if len(edge_set) != 48:
        raise ValueError("edge_index does not contain 48 unique directed edges")
    for source, destination in edge_set:
        sr, sc = divmod(source, 4)
        dr, dc = divmod(destination, 4)
        if abs(sr - dr) + abs(sc - dc) != 1:
            raise ValueError(f"non-cardinal physical edge {source}->{destination}")
        if (destination, source) not in edge_set:
            raise ValueError(f"missing reverse edge for {source}->{destination}")
    return {
        "shape": [2, 48],
        "directed_edges": 48,
        "undirected_links": 24,
        "physical_self_loops": 0,
    }


class CleanSourceOnlyBaseline(nn.Module):
    """Clean source-only model extracted from a seeded frozen B3 reference."""

    def __init__(
        self,
        reference_b3: nn.Module,
        operator: str,
        base_edge_index: torch.Tensor,
    ) -> None:
        super().__init__()
        if operator not in OPERATORS:
            raise ValueError(f"unsupported operator: {operator}")

        self.operator = operator
        self.input_projection = copy.deepcopy(reference_b3.input_projection)
        self.temporal_blocks = copy.deepcopy(reference_b3.temporal_blocks)
        self.node_projection = copy.deepcopy(reference_b3.node_projection)
        self.source_head = copy.deepcopy(reference_b3.source_head)
        self.register_buffer(
            "base_edge_index",
            base_edge_index.detach().cpu().long().contiguous(),
        )

        if operator == "conv1d":
            self.graph1 = None
            self.graph2 = None
        elif operator == "gcnconv":
            self.graph1 = GCNConv(64, 64, add_self_loops=True, normalize=True)
            self.graph2 = GCNConv(64, 64, add_self_loops=True, normalize=True)
        elif operator == "graphconv":
            self.graph1 = GraphConv(64, 64, aggr="add")
            self.graph2 = GraphConv(64, 64, aggr="add")
        elif operator == "gatconv":
            self.graph1 = GATConv(
                64,
                64,
                heads=1,
                concat=False,
                negative_slope=0.2,
                dropout=0.0,
                add_self_loops=True,
                bias=True,
            )
            self.graph2 = GATConv(
                64,
                64,
                heads=1,
                concat=False,
                negative_slope=0.2,
                dropout=0.0,
                add_self_loops=True,
                bias=True,
            )
        self.activation = nn.ReLU()

    def encode_nodes(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (16, 58, 32):
            raise ValueError(f"x shape={tuple(x.shape)}, expected [B,16,58,32]")
        if (
            physical_port_mask.ndim != 3
            or tuple(physical_port_mask.shape[1:]) != (16, 10)
            or physical_port_mask.shape[0] != x.shape[0]
        ):
            raise ValueError("physical_port_mask must have shape [B,16,10]")

        batch_size, num_nodes, num_features, time_steps = x.shape
        y = x.reshape(batch_size * num_nodes, num_features, time_steps)
        y = F.relu(self.input_projection(y))
        for block in self.temporal_blocks:
            y = block(y)
        temporal_embedding = y[:, :, -1].reshape(batch_size, num_nodes, 64)
        node_input = torch.cat(
            (
                temporal_embedding,
                physical_port_mask.to(dtype=temporal_embedding.dtype),
            ),
            dim=-1,
        )
        return F.relu(self.node_projection(node_input))

    def source_logits_without_graph(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.source_head(
            self.encode_nodes(x, physical_port_mask)
        ).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_embedding = self.encode_nodes(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
        if self.operator != "conv1d":
            flat = node_embedding.reshape(batch_size * num_nodes, embedding_dim)
            edges = batched_edge_index(
                self.base_edge_index,
                batch_size,
                num_nodes,
                flat.device,
            )
            flat = self.activation(self.graph1(flat, edges))
            flat = self.activation(self.graph2(flat, edges))
            node_embedding = flat.reshape(batch_size, num_nodes, embedding_dim)
        logits = self.source_head(node_embedding)
        if logits.ndim != 3 or logits.shape[-1] != 1:
            raise RuntimeError(f"source head returned unexpected shape {tuple(logits.shape)}")
        return logits.squeeze(-1)


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def train_one_epoch(
    *,
    model: CleanSourceOnlyBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source_pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.train()
    item_count = 0
    weighted_loss = 0.0
    maximum_preclip_gradient_norm = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        batch_items = int(batch["x"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["x"], batch["physical_port_mask"])
        target = batch["y_source"].float()
        if logits.shape != target.shape:
            raise RuntimeError("source logits/target shape mismatch")
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=source_pos_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRADIENT_CLIP
        )
        maximum_preclip_gradient_norm = max(
            maximum_preclip_gradient_norm,
            float(gradient_norm.item()),
        )
        optimizer.step()
        item_count += batch_items
        weighted_loss += float(loss.item()) * batch_items

    if item_count != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(
            f"training epoch consumed {item_count} items; expected {EXPECTED_TRAIN_ITEMS}"
        )
    return {
        "loss": weighted_loss / item_count,
        "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
    }


def validate(
    *,
    model: CleanSourceOnlyBaseline,
    loader: DataLoader,
    device: torch.device,
    source_pos_weight: torch.Tensor,
    return_predictions: bool,
) -> dict[str, Any]:
    model.eval()
    item_count = 0
    weighted_loss = 0.0
    truth_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_items = int(batch["x"].shape[0])
            logits = model(batch["x"], batch["physical_port_mask"])
            target = batch["y_source"].float()
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=source_pos_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            item_count += batch_items
            weighted_loss += float(loss.item()) * batch_items
            truth_parts.append(target.detach().cpu().numpy().astype(np.int64))
            score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())

    if item_count != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation consumed {item_count} items; expected {EXPECTED_VALIDATION_ITEMS}"
        )
    truth = np.concatenate(truth_parts, axis=0)
    score = np.concatenate(score_parts, axis=0)
    metrics = source_and_graph_metrics(truth, score, threshold=0.5)
    selection_score = checkpoint_selection_score(metrics)
    result: dict[str, Any] = {
        "loss": weighted_loss / item_count,
        "selection_score": selection_score,
        **metrics,
    }
    if return_predictions:
        result["truth"] = truth
        result["score"] = score
    return result


def checkpoint_rank(
    validation_metrics: dict[str, Any],
    epoch: int,
) -> tuple[float, float, float, float, float, int]:
    return (
        float(validation_metrics["selection_score"]),
        float(validation_metrics["source"]["average_precision"]),
        float(validation_metrics["source"]["auroc"]),
        float(validation_metrics["derived_graph"]["average_precision"]),
        -float(validation_metrics["loss"]),
        -int(epoch),
    )


def operator_configuration(operator: str) -> dict[str, Any]:
    if operator == "conv1d":
        return {"message_passing": False, "graph_layers": 0}
    if operator == "gcnconv":
        return {
            "message_passing": True,
            "graph_layers": 2,
            "operator": "GCNConv",
            "self_loops": True,
            "normalization": "symmetric",
            "edge_weights": "unit",
            "bias": True,
        }
    if operator == "graphconv":
        return {
            "message_passing": True,
            "graph_layers": 2,
            "operator": "GraphConv",
            "aggregation": "add",
            "stored_physical_self_loops": False,
            "explicit_root_transform": True,
            "bias": True,
        }
    if operator == "gatconv":
        return {
            "message_passing": True,
            "graph_layers": 2,
            "operator": "GATConv",
            "heads": 1,
            "concat": False,
            "self_loops": True,
            "negative_slope": 0.2,
            "attention_dropout": 0.0,
            "bias": True,
            "screening_only": True,
        }
    raise ValueError(f"unknown operator: {operator}")


def analytic_operation_count(operator: str) -> dict[str, int | str]:
    # Multiply-accumulate estimates per one 16-router sample.
    input_projection = 16 * 32 * 58 * 64
    temporal_block = 16 * 32 * 64 * 3 + 16 * 32 * 64 * 64
    temporal_blocks = 4 * temporal_block
    node_projection = 16 * 74 * 64
    source_head = 16 * (64 * 32 + 32)
    base_macs = input_projection + temporal_blocks + node_projection + source_head

    graph_linear_macs = 0
    graph_message_scalar_ops = 0
    if operator == "gcnconv":
        graph_linear_macs = 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * (48 + 16) * 64
    elif operator == "graphconv":
        graph_linear_macs = 2 * 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * 48 * 64
    elif operator == "gatconv":
        graph_linear_macs = 2 * 16 * 64 * 64
        graph_message_scalar_ops = 2 * (
            2 * 16 * 64 + (48 + 16) * (2 + 2 * 64)
        )

    return {
        "method": "analytic_per_sample_proxy",
        "base_temporal_node_head_macs": base_macs,
        "graph_linear_macs": graph_linear_macs,
        "graph_message_scalar_ops": graph_message_scalar_ops,
        "total_linear_macs": base_macs + graph_linear_macs,
    }


def measure_inference_latency(
    model: CleanSourceOnlyBaseline,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, float | int]:
    model.eval()
    x = batch["x"].to(device)
    mask = batch["physical_port_mask"].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    batch_items = int(x.shape[0])
    milliseconds_per_batch = 1000.0 * elapsed / repetitions
    return {
        "batch_items": batch_items,
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "milliseconds_per_batch": milliseconds_per_batch,
        "microseconds_per_item": 1000.0 * milliseconds_per_batch / batch_items,
    }


def parse_distribution(value: dict[str, Any]) -> dict[int, int]:
    return {int(key): int(frequency) for key, frequency in value.items()}


def role_positive_count(distribution: dict[int, int]) -> int:
    return sum(role_count * frequency for role_count, frequency in distribution.items())


def verify_contracts(args: argparse.Namespace) -> dict[str, Any]:
    g0 = args.g0_dir.resolve()
    smoke = args.smoke_dir.resolve()
    topology = args.topology_dir.resolve()
    b0_r3 = args.b0_r3_dir.resolve()

    paths = {
        "operator_contract": g0 / "V5_P2_G0_GRAPH_OPERATOR_CONTRACTS.json",
        "training_policy": g0 / "V5_P2_G0_TRAINING_SELECTION_AND_SEED_POLICY.json",
        "task_matrix": g0 / "V5_P2_G0_TASK_AND_MODEL_MATRIX.json",
        "test_policy": g0 / "V5_P2_G0_P2_TEST_ACCESS_POLICY.json",
        "smoke_contract": smoke / "G1_SOURCE_ONLY_GRAPH_SMOKE_CONTRACT.json",
        "smoke_lock": smoke / "G1_SOURCE_ONLY_GRAPH_SMOKE_CHECK_LOCK.json",
        "topology_report": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT.json",
        "topology_lock": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_TOPOLOGY_CONTRACT_LOCK.json",
        "edge_index": topology / "V5_P2_G1A_R2A_CANONICAL_STATIC_EDGE_INDEX.npy",
        "b0_r3_report": b0_r3 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT.json",
        "b0_r3_lock": b0_r3 / "V5_P2_B0_R3_CORRECTED_NONTEST_LABEL_AND_SHORTCUT_AUDIT_LOCK.json",
        "loader": args.loader_path.resolve(),
        "b3_model": args.b3_model_path.resolve(),
        "pair_manifest": args.pair_manifest.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing prerequisite {name}: {path}")

    if sha256_file(paths["loader"]) != EXPECTED_LOADER_SHA:
        raise RuntimeError("frozen P2 loader SHA changed")
    if sha256_file(paths["b3_model"]) != EXPECTED_B3_MODEL_SHA:
        raise RuntimeError("frozen B3 model SHA changed")
    if sha256_file(paths["edge_index"]) != EXPECTED_EDGE_SHA:
        raise RuntimeError("canonical edge_index SHA changed")

    operator_contract = load_json(paths["operator_contract"])
    training_policy = load_json(paths["training_policy"])
    task_matrix = load_json(paths["task_matrix"])
    test_policy = load_json(paths["test_policy"])
    smoke_contract = load_json(paths["smoke_contract"])
    smoke_lock = load_json(paths["smoke_lock"])
    topology_report = load_json(paths["topology_report"])
    topology_lock = load_json(paths["topology_lock"])
    b0_report = load_json(paths["b0_r3_report"])
    b0_lock = load_json(paths["b0_r3_lock"])

    if smoke_lock.get("status") != EXPECTED_SMOKE_STATUS:
        raise RuntimeError("G1 smoke is not complete")
    if smoke_lock.get("contract_sha256") != EXPECTED_SMOKE_CONTRACT_SHA:
        raise RuntimeError("G1 smoke contract SHA changed")
    if smoke_contract.get("contract_sha256") != EXPECTED_SMOKE_CONTRACT_SHA:
        raise RuntimeError("G1 smoke contract content changed")
    if smoke_contract.get("input_shape") != ["items", 16, 58, 32]:
        raise RuntimeError("G1 input shape changed")
    if smoke_contract.get("operators") != list(OPERATORS):
        raise RuntimeError("G1 operator order changed")
    if smoke_contract.get("graph_layers") != 2 or smoke_contract.get("graph_width") != 64:
        raise RuntimeError("G1 graph width/layer contract changed")
    if smoke_contract.get("production_constraints") != {
        "clean_encoder_equivalence_required": True,
        "clean_temporal_encoder_extraction_required": True,
        "hook_based_full_b3_wrapper_allowed": False,
        "unused_multitask_heads_allowed": False,
    }:
        raise RuntimeError("G1 production constraints changed")

    if operator_contract.get("common_architecture", {}).get("graph_layer_count") != 2:
        raise RuntimeError("G0 graph layer count changed")
    base = training_policy.get("training_base", {})
    expected_base = {
        "amp": False,
        "checkpoint_selection_split": "validation_only",
        "early_stopping_patience": 12,
        "learning_rate": 0.001,
        "max_epochs": 100,
        "minimum_epochs": 15,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "threshold_tuning_during_training": False,
        "weight_decay": 0.0001,
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise RuntimeError(f"G0 training field {key} changed")
    if base.get("initial_screening_seeds") != list(SEEDS):
        raise RuntimeError("G0 initial seeds changed")

    source_task = training_policy.get("task_selection", {}).get(
        "A_SOURCE_ONLY_LOCALIZATION", {}
    )
    if source_task.get("checkpoint_score") != {
        "derived_graph_average_precision": 0.25,
        "source_auroc": 0.25,
        "source_average_precision": 0.5,
    }:
        raise RuntimeError("source checkpoint score changed")
    if source_task.get("post_checkpoint_threshold_objective") != {
        "source_exact_set_attack": 0.4,
        "source_node_f1": 0.6,
    }:
        raise RuntimeError("source threshold objective changed")

    task_a = next(
        task for task in task_matrix.get("tasks", [])
        if task.get("task_id") == "A_SOURCE_ONLY_LOCALIZATION"
    )
    if task_a.get("participating_losses") != ["source_weighted_bce"]:
        raise RuntimeError("G1 participating loss changed")
    if test_policy.get("g1_to_g6", {}).get("p2_test_tensor_access_allowed") is not False:
        raise RuntimeError("test-access policy unexpectedly changed")

    canonical = topology_report.get("canonical_contract", {})
    if canonical.get("contract_sha256") != EXPECTED_TOPOLOGY_CONTRACT_SHA:
        raise RuntimeError("topology contract SHA changed")
    if topology_lock.get("edge_index_sha256") != EXPECTED_EDGE_SHA:
        raise RuntimeError("topology lock edge SHA changed")
    if b0_report.get("status") != "COMPLETE":
        raise RuntimeError("B0-R3 report is not complete")
    if b0_lock.get("report_sha256") != sha256_file(paths["b0_r3_report"]):
        raise RuntimeError("B0-R3 report SHA mismatch")
    if b0_lock.get("shortcut_block_count") != 0:
        raise RuntimeError("B0-R3 shortcut block count is not zero")
    if b0_lock.get("label_integrity_pass") is not True:
        raise RuntimeError("B0-R3 label integrity did not pass")

    edge_index = torch.from_numpy(np.load(paths["edge_index"], allow_pickle=False)).long()
    edge_runtime = validate_edge_index(edge_index)
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "edge_index": edge_index,
        "edge_runtime": edge_runtime,
        "training_policy": training_policy,
        "operator_contract": operator_contract,
        "task_matrix": task_matrix,
        "test_policy": test_policy,
        "smoke_contract": smoke_contract,
        "b0_report": b0_report,
    }


def write_hold(
    report_dir: Path,
    model_dir: Path,
    operator: str,
    seed: int,
    failure: str,
) -> None:
    report = {
        "stage": STAGE,
        "status": "HOLD",
        "operator": operator,
        "seed": seed,
        "failure": failure,
        "training_started": (
            report_dir / f"{STAGE}_TRAINING_STARTED"
        ).is_file(),
        "scientific_checkpoint_created": any(model_dir.glob("*.pt"))
        if model_dir.is_dir()
        else False,
        "p2_test_directory_enumerated": False,
        "p2_test_tensors_deserialized": False,
        "architecture_selected": False,
    }
    write_json(report_dir / f"{STAGE}.json", report)
    atomic_write(report_dir / HOLD, HOLD + "\n")


def run(args: argparse.Namespace) -> int:
    operator = args.operator
    seed = int(args.seed)
    model_dir = args.model_dir.resolve()
    report_dir = args.report_dir.resolve()

    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if PYG_IMPORT_ERROR is not None and operator != "conv1d":
        raise RuntimeError(f"PyTorch Geometric unavailable: {PYG_IMPORT_ERROR}")
    if model_dir.exists() or report_dir.exists():
        raise FileExistsError("model/report destination already exists")
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    contracts = verify_contracts(args)
    preflight_dir = args.preflight_dir.resolve()
    preflight_lock_path = (
        preflight_dir
        / "V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_LOCK.json"
    )
    preflight_complete = (
        preflight_dir
        / "V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
    )
    if not preflight_lock_path.is_file() or not preflight_complete.is_file():
        raise FileNotFoundError("clean G1C implementation preflight is not complete")
    preflight_lock = load_json(preflight_lock_path)
    if (
        preflight_lock.get("status")
        != "V5_P2_G1C_CLEAN_TRAINING_IMPLEMENTATION_PREFLIGHT_COMPLETE"
    ):
        raise RuntimeError("clean G1C implementation preflight status changed")
    if preflight_lock.get("scientific_training_authorized") is not True:
        raise RuntimeError("clean G1C implementation did not authorize training")
    if preflight_lock.get("trainer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("trainer source changed after clean preflight")
    if preflight_lock.get("edge_index_sha256") != EXPECTED_EDGE_SHA:
        raise RuntimeError("preflight edge_index SHA changed")
    set_seed(seed)

    loader_module = import_module(args.loader_path.resolve(), f"g1c_loader_{operator}_{seed}")
    b3_module = import_module(args.b3_model_path.resolve(), f"g1c_b3_{operator}_{seed}")
    DatasetClass = loader_module.V5P2PairAlignedPrimary58Dataset
    B3Class = b3_module.P2B3Conv1DOnlyCount4

    train_dataset = DatasetClass(
        root=args.root.resolve(),
        split="train",
        pair_manifest=args.pair_manifest.resolve(),
        window=32,
        stride=8,
    )
    validation_dataset = DatasetClass(
        root=args.root.resolve(),
        split="validation",
        pair_manifest=args.pair_manifest.resolve(),
        window=32,
        stride=8,
    )
    if len(train_dataset) != EXPECTED_TRAIN_ITEMS:
        raise RuntimeError(f"train length={len(train_dataset)}, expected {EXPECTED_TRAIN_ITEMS}")
    if len(validation_dataset) != EXPECTED_VALIDATION_ITEMS:
        raise RuntimeError(
            f"validation length={len(validation_dataset)}, expected {EXPECTED_VALIDATION_ITEMS}"
        )

    b0_train = contracts["b0_report"]["label_summaries"]["train"]
    source_distribution = parse_distribution(
        b0_train["active_source_count_distribution"]
    )
    positive_entries = role_positive_count(source_distribution)
    total_entries = EXPECTED_TRAIN_ITEMS * 16
    negative_entries = total_entries - positive_entries
    raw_pos_weight = negative_entries / positive_entries
    clamped_pos_weight = min(20.0, max(1.0, raw_pos_weight))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_sampler = PairBlockBatchSampler(
        train_dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )
    validation_sampler = PairBlockBatchSampler(
        validation_dataset,
        block_batch_size=PAIR_BLOCK_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )

    reference_b3 = B3Class()
    if sum(parameter.numel() for parameter in reference_b3.parameters()) != EXPECTED_FULL_B3_PARAMETERS:
        raise RuntimeError("frozen B3 parameter count changed")
    reference_b3_initial_sha = sha256_state_dict(reference_b3.state_dict())

    model = CleanSourceOnlyBaseline(
        reference_b3=reference_b3,
        operator=operator,
        base_edge_index=contracts["edge_index"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_CLEAN_PARAMETERS[operator]:
        raise RuntimeError(
            f"clean {operator} parameter count={parameter_count}, "
            f"expected {EXPECTED_CLEAN_PARAMETERS[operator]}"
        )

    # Clean extraction equivalence: no hooks and no retained unused heads.
    equivalence_items = [train_dataset[index] for index in (0, 1)]
    equivalence_batch = {
        key: torch.stack([item[key] for item in equivalence_items], dim=0)
        for key in equivalence_items[0]
    }
    reference_b3.eval()
    model.eval()
    with torch.no_grad():
        reference_logits = reference_b3(
            equivalence_batch["x"], equivalence_batch["physical_port_mask"]
        )["source_logits"]
        clean_logits = model.source_logits_without_graph(
            equivalence_batch["x"], equivalence_batch["physical_port_mask"]
        )
    equivalence_max_abs_error = float((reference_logits - clean_logits).abs().max())
    if equivalence_max_abs_error != 0.0:
        raise RuntimeError(
            f"clean encoder/source-head equivalence failed: {equivalence_max_abs_error}"
        )
    reference_source_head_sha = sha256_state_dict(reference_b3.source_head.state_dict())
    clean_source_head_sha = sha256_state_dict(model.source_head.state_dict())
    if reference_source_head_sha != clean_source_head_sha:
        raise RuntimeError("clean source head does not match seeded B3 source head")
    del reference_b3

    model = model.to(device)
    initial_state_sha = sha256_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-5,
    )
    source_pos_weight = torch.tensor(
        clamped_pos_weight,
        dtype=torch.float32,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = model_dir / f"v5_p2_g1c_{operator}_seed_{seed}_best.pt"
    history_path = report_dir / f"V5_P2_G1C_{operator.upper()}_SEED_{seed}_HISTORY.csv"
    progress_path = report_dir / f"V5_P2_G1C_{operator.upper()}_SEED_{seed}_PROGRESS.json"
    predictions_path = report_dir / f"V5_P2_G1C_{operator.upper()}_SEED_{seed}_VALIDATION_PREDICTIONS.npz"
    threshold_csv_path = report_dir / f"V5_P2_G1C_{operator.upper()}_SEED_{seed}_THRESHOLD_SWEEP.csv"

    best_rank: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_checkpoint_sha: str | None = None
    best_early_stop_score = -float("inf")
    early_stop_counter = 0
    stopped_early = False
    history_rows: list[dict[str, Any]] = []
    start_time = time.time()

    print("===== V5 P2-G1C SOURCE-ONLY SINGLE RUN =====")
    print(f"operator: {operator}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"train_items: {len(train_dataset)}")
    print(f"validation_items: {len(validation_dataset)}")
    print(f"train_batches: {len(train_loader)}")
    print(f"validation_batches: {len(validation_loader)}")
    print(f"parameter_count: {parameter_count}")
    print(f"source_pos_weight: {clamped_pos_weight}")
    print("test_directory_enumerated: false")
    print("test_tensors_deserialized: false")
    atomic_write(
        report_dir / f"{STAGE}_TRAINING_STARTED",
        f"{STAGE}_TRAINING_STARTED\n",
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            source_pos_weight=source_pos_weight,
        )
        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            source_pos_weight=source_pos_weight,
            return_predictions=False,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        candidate_rank = checkpoint_rank(validation_metrics, epoch)
        is_best = best_rank is None or candidate_rank > best_rank
        if is_best:
            best_rank = candidate_rank
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            checkpoint_payload = {
                "stage": STAGE,
                "operator": operator,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "initial_state_sha256": initial_state_sha,
                "reference_b3_initial_state_sha256": reference_b3_initial_sha,
                "source_positive_weight": clamped_pos_weight,
                "validation_metrics": validation_metrics,
                "contract_sha256": EXPECTED_SMOKE_CONTRACT_SHA,
                "edge_index_sha256": EXPECTED_EDGE_SHA,
                "test_tensors_deserialized": False,
            }
            atomic_torch_save(checkpoint_payload, checkpoint_path)
            best_checkpoint_sha = sha256_file(checkpoint_path)

        current_score = float(validation_metrics["selection_score"])
        if current_score > best_early_stop_score + EARLY_STOP_MIN_DELTA:
            best_early_stop_score = current_score
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        scheduler.step(current_score)

        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "maximum_preclip_gradient_norm": train_metrics[
                "maximum_preclip_gradient_norm"
            ],
            "validation_loss": validation_metrics["loss"],
            "validation_selection_score": current_score,
            "source_average_precision": validation_metrics["source"][
                "average_precision"
            ],
            "source_auroc": validation_metrics["source"]["auroc"],
            "source_fixed_0_5_f1": validation_metrics["source"][
                "thresholded"
            ]["f1"],
            "derived_graph_average_precision": validation_metrics[
                "derived_graph"
            ]["average_precision"],
            "derived_graph_auroc": validation_metrics["derived_graph"]["auroc"],
            "derived_graph_fixed_0_5_balanced_accuracy": validation_metrics[
                "derived_graph"
            ]["thresholded"]["balanced_accuracy"],
            "exact_set_attack_fixed_0_5": validation_metrics["exact_set"]["attack"],
            "is_best_checkpoint": is_best,
            "early_stop_patience_counter": early_stop_counter,
            "elapsed_seconds": elapsed,
        }
        history_rows.append(row)
        write_csv(history_path, history_rows)
        write_json(
            progress_path,
            {
                "stage": STAGE,
                "status": "RUNNING",
                "operator": operator,
                "seed": seed,
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": best_checkpoint_sha,
                "current_validation_metrics": validation_metrics,
                "early_stop_patience_counter": early_stop_counter,
                "current_learning_rate_after_scheduler": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "p2_test_directory_enumerated": False,
                "p2_test_tensors_deserialized": False,
            },
        )
        print(
            f"operator={operator} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"score={current_score:.6f} "
            f"src_ap={validation_metrics['source']['average_precision']:.6f} "
            f"src_auc={validation_metrics['source']['auroc']:.6f} "
            f"graph_ap={validation_metrics['derived_graph']['average_precision']:.6f} "
            f"lr={learning_rate:.8g} best_epoch={best_epoch} "
            f"patience={early_stop_counter}"
        )
        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            stopped_early = True
            print(f"early stopping at epoch {epoch}; patience reached {EARLY_STOP_PATIENCE}")
            break

    if best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("no best checkpoint was created")
    if best_checkpoint_sha != sha256_file(checkpoint_path):
        raise RuntimeError("best checkpoint SHA changed")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final_validation = validate(
        model=model,
        loader=validation_loader,
        device=device,
        source_pos_weight=source_pos_weight,
        return_predictions=True,
    )
    truth = final_validation.pop("truth")
    score = final_validation.pop("score")
    threshold_result = tune_source_threshold(truth, score)
    write_csv(threshold_csv_path, threshold_result.pop("rows"))
    np.savez_compressed(
        predictions_path,
        y_source=truth.astype(np.uint8),
        source_score=score.astype(np.float32),
    )

    representative_batch = next(iter(validation_loader))
    latency = measure_inference_latency(model, representative_batch, device)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    total_elapsed = time.time() - start_time

    selected_threshold = threshold_result["selected"]
    run_core = {
        "stage": STAGE,
        "status": "COMPLETE",
        "operator": operator,
        "seed": seed,
        "architecture": {
            "name": f"CLEAN_B3_SOURCE_ONLY_{operator.upper()}",
            "parameter_count": parameter_count,
            "expected_parameter_count": EXPECTED_CLEAN_PARAMETERS[operator],
            "graph_layers": 0 if operator == "conv1d" else 2,
            "graph_width": 64,
            "operator_configuration": operator_configuration(operator),
            "activation": "ReLU",
            "dropout": 0.0,
            "residual": False,
            "initial_state_sha256": initial_state_sha,
            "reference_b3_initial_state_sha256": reference_b3_initial_sha,
            "reference_source_head_sha256": reference_source_head_sha,
            "clean_source_head_sha256": clean_source_head_sha,
            "clean_equivalence_max_abs_error": equivalence_max_abs_error,
            "unused_multitask_heads_present": False,
            "hook_based_extraction_used": False,
        },
        "data": {
            "train_items": len(train_dataset),
            "validation_items": len(validation_dataset),
            "item_batch_size": ITEM_BATCH_SIZE,
            "pair_block_batch_size": PAIR_BLOCK_BATCH_SIZE,
            "train_batches": len(train_loader),
            "validation_batches": len(validation_loader),
            "num_workers": 0,
            "window": 32,
            "stride": 8,
            "input_shape": ["items", 16, 58, 32],
            "pair_manifest": str(args.pair_manifest.resolve()),
        },
        "loss": {
            "name": "class_weighted_source_bce_with_logits",
            "positive_entries": positive_entries,
            "negative_entries": negative_entries,
            "raw_positive_weight": raw_pos_weight,
            "clamped_positive_weight": clamped_pos_weight,
        },
        "training": {
            "device": str(device),
            "maximum_epochs": MAX_EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "completed_epoch": len(history_rows),
            "stopped_early": stopped_early,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "early_stopping_minimum_delta": EARLY_STOP_MIN_DELTA,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 4,
                "threshold": 1e-4,
                "threshold_mode": "abs",
                "minimum_learning_rate": 1e-5,
            },
            "gradient_clip_global_norm": GRADIENT_CLIP,
            "automatic_mixed_precision": False,
            "deterministic_algorithms": "enabled_warn_only",
            "total_elapsed_seconds": total_elapsed,
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "checkpoint_selection_weights": CHECKPOINT_WEIGHTS,
            "validation_metrics_at_selection": best_validation_metrics,
            "path": str(checkpoint_path),
            "sha256": best_checkpoint_sha,
            "ranking_tuple": list(best_rank) if best_rank is not None else None,
        },
        "post_checkpoint_validation": {
            "untuned_metrics": final_validation,
            "threshold_selection": threshold_result,
        },
        "cost": {
            "operation_count": analytic_operation_count(operator),
            "peak_cuda_memory_allocated_bytes": peak_memory,
            "inference_latency_proxy": latency,
        },
        "artifacts": {
            "history_csv": [str(history_path), sha256_file(history_path)],
            "progress_json": [str(progress_path), sha256_file(progress_path)],
            "checkpoint": [str(checkpoint_path), sha256_file(checkpoint_path)],
            "validation_predictions": [
                str(predictions_path),
                sha256_file(predictions_path),
            ],
            "threshold_sweep_csv": [
                str(threshold_csv_path),
                sha256_file(threshold_csv_path),
            ],
        },
        "contracts": {
            "smoke_contract_sha256": EXPECTED_SMOKE_CONTRACT_SHA,
            "topology_contract_sha256": EXPECTED_TOPOLOGY_CONTRACT_SHA,
            "edge_index_sha256": EXPECTED_EDGE_SHA,
            "loader_sha256": EXPECTED_LOADER_SHA,
            "b3_model_sha256": EXPECTED_B3_MODEL_SHA,
        },
        "security_boundary": {
            "p2_train_used": True,
            "p2_validation_used": True,
            "p2_test_directory_existence_checked": False,
            "p2_test_directory_enumerated": False,
            "p2_test_tensor_files_opened": False,
            "p2_test_tensors_deserialized": False,
            "p2_test_evaluation_performed": False,
            "threshold_tuning_during_training": False,
            "post_checkpoint_validation_threshold_tuning": True,
            "architecture_selected": False,
            "quantization_performed": False,
            "rtl_generated": False,
        },
        "next_stage": "V5_P2_G1C_SOURCE_ONLY_GRAPH_MATRIX_AGGREGATION",
    }
    run_report = {**run_core, "run_contract_sha256": canonical_sha256(run_core)}
    report_path = report_dir / f"{STAGE}.json"
    write_json(report_path, run_report)
    atomic_write(report_dir / COMPLETE, COMPLETE + "\n")
    print(COMPLETE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--g0-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--b0-r3-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--loader-path", type=Path, required=True)
    parser.add_argument("--b3-model-path", type=Path, required=True)
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.expanduser().resolve()
    operator = str(args.operator)
    seed = int(args.seed)
    try:
        return run(args)
    except Exception as exc:
        report_dir.mkdir(parents=True, exist_ok=True)
        failure = f"{type(exc).__name__}: {exc}"
        write_hold(
            report_dir,
            args.model_dir.expanduser().resolve(),
            operator,
            seed,
            failure,
        )
        print(HOLD)
        print("FAIL:", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
