#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np

def load_graph_arrays(data_path):
    data_path = Path(data_path)

    if data_path.is_dir():
        print("loading dataset folder:", data_path)
        out = {
            "x": np.load(data_path / "x.npy", mmap_mode="r"),
            "y_graph": np.load(data_path / "y_graph.npy", mmap_mode="r"),
            "y_node": np.load(data_path / "y_node.npy", mmap_mode="r"),
            "edge_index": np.load(data_path / "edge_index.npy"),
            "run_id": np.load(data_path / "run_id.npy", allow_pickle=True),
            "end_epoch": np.load(data_path / "end_epoch.npy", mmap_mode="r"),
        }

        for name in ["attackers", "split", "feature_cols", "profile", "active_cores", "strength", "seed"]:
            f = data_path / f"{name}.npy"
            out[name] = np.load(f, allow_pickle=True) if f.exists() else None

        return out

    print("loading npz dataset:", data_path)
    d = np.load(data_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_v3_splits(data_path):
    d = load_graph_arrays(data_path)

    if d.get("split") is None:
        raise RuntimeError("V3 split requested, but dataset has no split.npy / split field.")

    split = d["split"].astype(str)

    train_idx = np.where(split == "train")[0].astype(np.int64)
    val_idx = np.where(split == "val")[0].astype(np.int64)
    test_idx = np.where(split == "test")[0].astype(np.int64)

    print()
    print("using internal V3 split")
    print("train:", len(train_idx))
    print("val:  ", len(val_idx))
    print("test: ", len(test_idx))

    return train_idx, val_idx, test_idx


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoCTemporalGraphDataset(Dataset):
    def __init__(self, npz_path: str, indices: np.ndarray):
        d = load_graph_arrays(npz_path)
        self.x = d["x"].astype(np.float32)
        self.y_graph = d["y_graph"].astype(np.float32)
        self.y_node = d["y_node"].astype(np.float32)
        self.run_id = d["run_id"].astype(str)
        self.end_epoch = d["end_epoch"].astype(np.int32)
        self.attackers = d["attackers"].astype(str)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return {
            "x": torch.from_numpy(self.x[real_idx]),
            "y_graph": torch.tensor(self.y_graph[real_idx], dtype=torch.float32),
            "y_node": torch.from_numpy(self.y_node[real_idx]),
            "run_id": self.run_id[real_idx],
            "end_epoch": int(self.end_epoch[real_idx]),
            "attackers": self.attackers[real_idx],
        }


def make_splits(npz_path: str, split_mode: str):
    d = load_graph_arrays(npz_path)
    run_ids = d["run_id"].astype(str)
    y_graph = d["y_graph"].astype(np.int64)
    unique_runs = sorted(set(run_ids.tolist()))

    print("Runs in dataset:")
    for run_id in unique_runs:
        count = int((run_ids == run_id).sum())
        first_index = np.where(run_ids == run_id)[0][0]
        label = int(y_graph[first_index])
        print(f"  {run_id:38s} samples={count} label={label}")

    if split_mode == "prototype":
        train_idx, val_idx, test_idx = [], [], []
        for run_id in unique_runs:
            idx = np.sort(np.where(run_ids == run_id)[0])
            n = len(idx)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    if split_mode == "placement":
        train_attack_runs = {
            "N-0-15-A-1-S20-V2",
            "N-0-15-A-7-S20-V2",
            "N-0-15-A-11-S20-V2",
            "N-0-15-A-1-7-S63-V2",
            "N-0-15-A-1-11-S63-V2",
            "N-0-15-A-7-12-S63-V2",
            "N-0-15-A-1-7-11-S77-V2",
            "N-0-15-A-1-7-12-S77-V2",
        }
        val_attack_runs = {
            "N-0-15-A-12-S20-V2",
            "N-0-15-A-11-12-S63-V2",
        }
        test_attack_runs = {
            "N-0-15-A-1-11-12-S77-V2",
            "N-0-15-A-7-11-12-S77-V2",
        }

        train_idx, val_idx, test_idx = [], [], []
        for run_id in unique_runs:
            idx = np.sort(np.where(run_ids == run_id)[0])
            if "-A-" not in run_id:
                n = len(idx)
                n_train = int(0.70 * n)
                n_val = int(0.15 * n)
                train_idx.extend(idx[:n_train])
                val_idx.extend(idx[n_train:n_train + n_val])
                test_idx.extend(idx[n_train + n_val:])
            elif run_id in train_attack_runs:
                train_idx.extend(idx)
            elif run_id in val_attack_runs:
                val_idx.extend(idx)
            elif run_id in test_attack_runs:
                test_idx.extend(idx)
            else:
                raise RuntimeError(f"Run {run_id!r} was not assigned to any split.")

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    raise ValueError(f"Unknown split mode: {split_mode}")


def build_normalized_adjacency(
    edge_index: np.ndarray,
    num_nodes: int,
) -> torch.Tensor:
    # Binary incoming-neighbor adjacency for GraphConv.
    # Self-loops are excluded because the root transform handles self.
    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float32,
    )

    sources = edge_index[0]
    targets = edge_index[1]

    for source, target in zip(sources, targets):
        source_i = int(source)
        target_i = int(target)
        if source_i == target_i:
            continue
        adjacency[target_i, source_i] = 1.0

    return adjacency


class GraphConvLayer(nn.Module):
    # Morris/PyG-style GraphConv with add aggregation:
    # h_out_i = W_root h_i + W_neighbor sum(j in N(i)) h_j + b.

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
        return (
            self.lin_neighbor(neighbor_sum)
            + self.lin_root(h)
        )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "acc": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def node_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    accuracy = accuracy_score(flat_true, flat_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average="binary", zero_division=0
    )
    exact_localization = np.all(y_pred == y_true, axis=1).mean()
    return {
        "node_acc": float(accuracy),
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
        "exact_localization": float(exact_localization),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    a_hat: torch.Tensor,
    device: torch.device,
    graph_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    graph_threshold: float = 0.5,
    node_threshold: float = 0.5,
):
    model.eval()
    graph_probs, graph_true = [], []
    node_probs, node_true = [], []
    total_loss = 0.0
    batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_graph = batch["y_graph"].to(device)
        y_node = batch["y_node"].to(device)
        graph_logits, node_logits = model(x, a_hat)

        graph_loss = graph_loss_fn(graph_logits, y_graph)
        node_loss = node_loss_fn(node_logits, y_node)
        loss = graph_loss + node_loss_weight * node_loss

        total_loss += float(loss.item())
        batches += 1
        graph_probs.append(torch.sigmoid(graph_logits).cpu().numpy())
        graph_true.append(y_graph.cpu().numpy())
        node_probs.append(torch.sigmoid(node_logits).cpu().numpy())
        node_true.append(y_node.cpu().numpy())

    graph_probs = np.concatenate(graph_probs)
    graph_true = np.concatenate(graph_true)
    node_probs = np.concatenate(node_probs)
    node_true = np.concatenate(node_true)

    graph_results = binary_metrics(graph_true, graph_probs, graph_threshold)
    node_results = node_metrics(node_true, node_probs, node_threshold)
    return {
        "loss": float(total_loss / max(1, batches)),
        **graph_results,
        **node_results,
    }


def print_metrics(prefix: str, metrics: dict) -> None:
    print(
        f"{prefix} "
        f"loss={metrics['loss']:.4f} "
        f"g_acc={metrics['acc']:.4f} "
        f"g_prec={metrics['precision']:.4f} "
        f"g_rec={metrics['recall']:.4f} "
        f"g_f1={metrics['f1']:.4f} "
        f"g_fpr={metrics['fpr']:.4f} "
        f"node_prec={metrics['node_precision']:.4f} "
        f"node_rec={metrics['node_recall']:.4f} "
        f"node_f1={metrics['node_f1']:.4f} "
        f"exact_loc={metrics['exact_localization']:.4f}"
    )


class TemporalEncoder(nn.Module):
    def __init__(self, in_features: int = 2, embedding_dim: int = 8, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * num_nodes, time_steps, feature_dim)
        x = x.permute(0, 2, 1)
        h = F.relu(self.conv(x))
        h = torch.max(h, dim=2).values
        return h.reshape(batch_size, num_nodes, -1)


class TemporalGCN(nn.Module):
    def __init__(self, input_features: int, temporal_dim: int = 8, gcn_hidden: int = 16, gcn_out: int = 8):
        super().__init__()
        self.temporal = TemporalEncoder(input_features, temporal_dim, kernel_size=3)
        self.gcn1 = GraphConvLayer(temporal_dim, gcn_hidden)
        self.gcn2 = GraphConvLayer(gcn_hidden, gcn_out)
        self.node_head = nn.Linear(gcn_out, 1)
        self.graph_head = nn.Linear(gcn_out, 1)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor):
        h = self.temporal(x)
        h = F.relu(self.gcn1(h, a_hat))
        h = F.relu(self.gcn2(h, a_hat))
        node_logits = self.node_head(h).squeeze(-1)
        graph_embedding = h.mean(dim=1)
        graph_logits = self.graph_head(graph_embedding).squeeze(-1)
        return graph_logits, node_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="graph_dataset/paper1_temporal_graphs_full.npz")
    parser.add_argument("--out-dir", default="models/temporal_gcn_conv1d_early_stop")
    parser.add_argument("--split-mode", choices=["prototype", "placement", "v3"], default="prototype")
    parser.add_argument("--splits-file", default=None, help="Optional .npz file containing train_idx, val_idx, test_idx.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-dim", type=int, default=8)
    parser.add_argument("--gcn-hidden", type=int, default=16)
    parser.add_argument("--gcn-out", type=int, default=8)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-threshold", type=float, default=0.5)
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative.")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data = load_graph_arrays(args.data)
    x_shape = data["x"].shape
    if len(x_shape) != 4:
        raise ValueError(f"Expected x shape [samples, nodes, time, features], received {x_shape}.")

    num_nodes = int(x_shape[1])
    input_feature_dim = int(x_shape[-1])
    edge_index = data["edge_index"].astype(np.int64)
    a_hat = build_normalized_adjacency(edge_index, num_nodes).to(device)

    if args.split_mode == "v3":
        train_idx, val_idx, test_idx = make_v3_splits(args.data)
    elif args.splits_file is not None:
        split_data = np.load(args.splits_file)
        train_idx = split_data["train_idx"].astype(np.int64)
        val_idx = split_data["val_idx"].astype(np.int64)
        test_idx = split_data["test_idx"].astype(np.int64)
        print("\nloaded external splits:", args.splits_file)
    else:
        train_idx, val_idx, test_idx = make_splits(args.data, args.split_mode)

    print("\nsplit sizes:")
    print("  train:", len(train_idx))
    print("  val:  ", len(val_idx))
    print("  test: ", len(test_idx))

    y_graph_train = data["y_graph"][train_idx].astype(np.float32)
    y_node_train = data["y_node"][train_idx].astype(np.float32)
    graph_pos = float(y_graph_train.sum())
    graph_neg = float(len(y_graph_train) - graph_pos)
    graph_pos_weight = graph_neg / max(graph_pos, 1.0)
    node_pos = float(y_node_train.sum())
    node_neg = float(y_node_train.size - node_pos)
    node_pos_weight = node_neg / max(node_pos, 1.0)

    print("\nclass weights:")
    print("  graph pos weight:", graph_pos_weight)
    print("  node pos weight: ", node_pos_weight)

    train_dataset = NoCTemporalGraphDataset(args.data, train_idx)
    val_dataset = NoCTemporalGraphDataset(args.data, val_idx)
    test_dataset = NoCTemporalGraphDataset(args.data, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("input feature dim:", input_feature_dim)
    print("number of nodes:", num_nodes)

    model = TemporalGCN(input_features=input_feature_dim, temporal_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, gcn_out=args.gcn_out).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("\nmodel:")
    print(model)
    print("parameters:", total_params)

    graph_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(graph_pos_weight, dtype=torch.float32, device=device)
    )
    node_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(node_pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)

        for batch in progress:
            x = batch["x"].to(device)
            y_graph = batch["y_graph"].to(device)
            y_node = batch["y_node"].to(device)
            graph_logits, node_logits = model(x, a_hat)
            graph_loss = graph_loss_fn(graph_logits, y_graph)
            node_loss = node_loss_fn(node_logits, y_node)
            loss = graph_loss + args.node_loss_weight * node_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_step_loss = total_loss / max(1, batches)
        eval_kwargs = {
            "a_hat": a_hat,
            "device": device,
            "graph_loss_fn": graph_loss_fn,
            "node_loss_fn": node_loss_fn,
            "node_loss_weight": args.node_loss_weight,
            "graph_threshold": args.graph_threshold,
            "node_threshold": args.node_threshold,
        }
        train_metrics = evaluate(model, train_loader, **eval_kwargs)
        val_metrics = evaluate(model, val_loader, **eval_kwargs)

        print(f"\nepoch {epoch:03d} train_step_loss={train_step_loss:.4f}")
        print_metrics("  train", train_metrics)
        print_metrics("  val  ", val_metrics)

        val_score = val_metrics["f1"] + val_metrics["node_f1"]
        previous_best = None if best_val_score == -float("inf") else float(best_val_score)
        improvement = val_score - best_val_score

        if improvement > args.min_delta:
            best_val_score = float(val_score)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "A_hat": a_hat.detach().cpu(),
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
                "val_metrics": val_metrics,
                "model_name": "Conv1D-GraphConv-TemporalGNN",
            }
            torch.save(checkpoint, out_dir / "best_model.pt")
            print(f"  validation score improved to {best_val_score:.6f}; saved best_model.pt")
        else:
            epochs_without_improvement += 1
            print(f"  no meaningful validation improvement ({epochs_without_improvement}/{args.patience})")

        history.append({
            "epoch": int(epoch),
            "train_loss_step": float(train_step_loss),
            "train": train_metrics,
            "val": val_metrics,
            "val_score": float(val_score),
            "best_val_score_before_epoch": previous_best,
            "best_val_score_after_epoch": float(best_val_score),
            "epochs_without_improvement": int(epochs_without_improvement),
        })

        with (out_dir / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"\nearly stopping triggered at epoch {stopped_epoch}. "
                f"Best epoch was {best_epoch} with validation score {best_val_score:.6f}."
            )
            break

    print("\nloading best checkpoint...")
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_kwargs = {
        "a_hat": a_hat,
        "device": device,
        "graph_loss_fn": graph_loss_fn,
        "node_loss_fn": node_loss_fn,
        "node_loss_weight": args.node_loss_weight,
        "graph_threshold": args.graph_threshold,
        "node_threshold": args.node_threshold,
    }
    train_metrics = evaluate(model, train_loader, **eval_kwargs)
    val_metrics = evaluate(model, val_loader, **eval_kwargs)
    test_metrics = evaluate(model, test_loader, **eval_kwargs)

    print("\nFINAL")
    print("model:", "Conv1D-GraphConv-TemporalGNN")
    print("best epoch:", checkpoint["best_epoch"])
    print("best validation score:", checkpoint["best_val_score"])
    print_metrics("train", train_metrics)
    print_metrics("val  ", val_metrics)
    print_metrics("test ", test_metrics)

    summary = {
        "model": "Conv1D-GraphConv-TemporalGNN",
        "data": args.data,
        "split_mode": args.split_mode,
        "splits_file": args.splits_file,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_score": float(checkpoint["best_val_score"]),
        "parameters": int(total_params),
        "input_feature_dim": int(input_feature_dim),
        "num_nodes": int(num_nodes),
        "training": {
            "maximum_epochs": int(args.epochs),
            "epochs_completed": int(len(history)),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "stopped_early": bool(stopped_early),
            "stopped_epoch": stopped_epoch,
            "seed": int(args.seed),
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_weights": {
            "graph_pos_weight": float(graph_pos_weight),
            "node_pos_weight": float(node_pos_weight),
        },
        "thresholds": {
            "graph": float(args.graph_threshold),
            "node": float(args.node_threshold),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    with (out_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        out_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print("\nwrote:")
    print(" ", out_dir / "best_model.pt")
    print(" ", out_dir / "history.json")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "splits.npz")


if __name__ == "__main__":
    main()
