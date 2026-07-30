#!/usr/bin/env python3
"""Frozen Task-D GraphConv challenger for the complete P2 multitask interface."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GraphConv
except Exception as exc:  # pragma: no cover - validated by preflight
    GraphConv = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


def batched_edge_index(
    base_edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge = base_edge_index.to(device=device, dtype=torch.long)
    if tuple(edge.shape) != (2, 48):
        raise ValueError(f"base_edge_index shape={tuple(edge.shape)}, expected [2,48]")
    offsets = (
        torch.arange(batch_size, device=device, dtype=torch.long)
        .view(batch_size, 1, 1)
        * num_nodes
    )
    result = (edge.view(1, 2, 48) + offsets).permute(1, 0, 2).reshape(2, -1)
    source_graph = torch.div(result[0], num_nodes, rounding_mode="floor")
    target_graph = torch.div(result[1], num_nodes, rounding_mode="floor")
    if not torch.equal(source_graph, target_graph):
        raise RuntimeError("batched edge_index contains cross-graph edges")
    return result


class P2TaskDGraphConvCount4(nn.Module):
    """
    B3 temporal encoder + two frozen-width GraphConv layers + unchanged B3 heads.

    Common B3 modules are deep-copied from a seeded reference B3 instance so
    the Conv1D and GraphConv candidates begin with byte-identical common
    parameters for every matched seed.
    """

    architecture_name = "TASK_D_B3_TEMPORAL_PLUS_2X_GRAPHCONV_COUNT4"
    expected_parameter_count = 59_785
    count_class_values = (1, 2, 3, 4)

    def __init__(self, reference_b3: nn.Module, base_edge_index: torch.Tensor) -> None:
        super().__init__()
        if GraphConv is None:
            raise RuntimeError(f"torch_geometric GraphConv unavailable: {_PYG_IMPORT_ERROR}")

        # Exact common B3 components.
        self.input_projection = copy.deepcopy(reference_b3.input_projection)
        self.temporal_blocks = copy.deepcopy(reference_b3.temporal_blocks)
        self.node_projection = copy.deepcopy(reference_b3.node_projection)
        self.source_head = copy.deepcopy(reference_b3.source_head)
        self.transit_head = copy.deepcopy(reference_b3.transit_head)
        self.victim_head = copy.deepcopy(reference_b3.victim_head)
        self.path_head = copy.deepcopy(reference_b3.path_head)
        self.graph_projection = copy.deepcopy(reference_b3.graph_projection)
        self.attack_head = copy.deepcopy(reference_b3.attack_head)
        self.count_head = copy.deepcopy(reference_b3.count_head)

        self.register_buffer(
            "base_edge_index",
            base_edge_index.detach().cpu().long().contiguous(),
        )
        self.graph1 = GraphConv(64, 64, aggr="add", bias=True)
        self.graph2 = GraphConv(64, 64, aggr="add", bias=True)
        self.activation = nn.ReLU()

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError(
                f"Task-D GraphConv parameter count={parameter_count}, "
                f"expected {self.expected_parameter_count}"
            )

    def encode_pre_graph(
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

    def forward(
        self,
        x: torch.Tensor,
        physical_port_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_embedding = self.encode_pre_graph(x, physical_port_mask)
        batch_size, num_nodes, embedding_dim = node_embedding.shape
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

        source_logits = self.source_head(node_embedding).squeeze(-1)
        transit_logits = self.transit_head(node_embedding).squeeze(-1)
        victim_logits = self.victim_head(node_embedding).squeeze(-1)
        path_logits = self.path_head(node_embedding).squeeze(-1)

        pooled = torch.cat(
            (node_embedding.mean(dim=1), node_embedding.amax(dim=1)),
            dim=-1,
        )
        graph_embedding = self.graph_projection(pooled)
        outputs = {
            "attack_logits": self.attack_head(graph_embedding).squeeze(-1),
            "count_logits": self.count_head(graph_embedding),
            "source_logits": source_logits,
            "transit_logits": transit_logits,
            "victim_logits": victim_logits,
            "path_logits": path_logits,
        }
        expected = {
            "attack_logits": (batch_size,),
            "count_logits": (batch_size, 4),
            "source_logits": (batch_size, 16),
            "transit_logits": (batch_size, 16),
            "victim_logits": (batch_size, 16),
            "path_logits": (batch_size, 16),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise RuntimeError(f"{key} shape={tuple(outputs[key].shape)}, expected {shape}")
        return outputs


def common_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only parameters/buffers shared by both Task-D candidates."""
    excluded_prefixes = ("graph1.", "graph2.", "base_edge_index")
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(excluded_prefixes)
    }


def operation_count_proxy(candidate: str) -> dict[str, int]:
    # Linear/depthwise MAC proxy for one [16,58,32] item.
    base = 10_899_776
    if candidate == "conv1d":
        return {
            "total_linear_macs": base,
            "graph_linear_macs": 0,
            "graph_message_scalar_ops": 0,
        }
    if candidate == "graphconv":
        graph_linear = 2 * 2 * 16 * 64 * 64
        return {
            "total_linear_macs": base + graph_linear,
            "graph_linear_macs": graph_linear,
            "graph_message_scalar_ops": 2 * 48 * 64,
        }
    raise ValueError(candidate)
