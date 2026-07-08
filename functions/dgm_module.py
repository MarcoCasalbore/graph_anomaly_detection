from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small feed-forward network used for node embeddings and classification."""

    def __init__(self, dims: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicGraphModule(nn.Module):
    """Minimal DGM-style latent kNN graph learner.

    The initial graph is used as a weak structural context, similarly to the
    W-DGM setting where a GCN/GAT embedding function receives an edge_index.
    The learned graph is then rebuilt from distances in the latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8, k: int = 3):
        super().__init__()
        self.embed = MLP([2 * input_dim, hidden_dim, latent_dim])
        self.k = int(k)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_logits = None
        self.last_edge_index = None
        self.stability_loss = torch.tensor(0.0)

    @staticmethod
    def aggregate_initial_neighbors(x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        """Mean aggregation over an initial weak graph, used as DGM context."""
        if edge_index is None:
            return torch.zeros_like(x)

        src, dst = edge_index
        out = torch.zeros_like(x)
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        out.index_add_(0, src, x[dst])
        deg.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(-1)

    def forward(self, x: torch.Tensor, initial_edge_index: torch.Tensor | None = None):
        context = self.aggregate_initial_neighbors(x, initial_edge_index)
        z = self.embed(torch.cat([x, context], dim=-1))

        diag_mask = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
        dist = torch.cdist(z, z, p=2).pow(2)
        scale = torch.exp(self.temperature.clamp(-5.0, 5.0))
        logits = -dist * scale
        logits = logits.masked_fill(diag_mask, -1e9)

        probs = torch.softmax(logits, dim=1)
        probs = probs.masked_fill(diag_mask, 0.0)

        k = min(max(1, self.k), x.shape[0] - 1)
        idx = torch.topk(probs, k=k, dim=1, largest=True).indices
        src = torch.arange(x.shape[0], device=x.device).view(-1, 1).expand_as(idx).reshape(-1)
        dst = idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

        self.last_logits = logits
        self.last_edge_index = edge_index
        self.stability_loss = torch.tensor(0.0, device=x.device)
        return z, edge_index, probs


class DGMNodeAnomalyModel(nn.Module):
    """Node-level anomaly detector with a DGM-style learned graph."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dgm = DynamicGraphModule(input_dim, hidden_dim, latent_dim, k=k)
        self.node_encoder = MLP([input_dim + latent_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    @staticmethod
    def aggregate_neighbors(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        out = torch.zeros_like(z)
        deg = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        out.index_add_(0, src, z[dst])
        deg.index_add_(0, src, torch.ones_like(src, dtype=z.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(-1)

    def forward(self, x: torch.Tensor, initial_edge_index: torch.Tensor | None = None):
        z, edge_index, edge_probs = self.dgm(x, initial_edge_index=initial_edge_index)
        neigh = self.aggregate_neighbors(z, edge_index)
        h = self.node_encoder(torch.cat([x, neigh], dim=-1))
        logits = self.classifier(h).squeeze(-1)
        return {
            "logits": logits,
            "embeddings": h,
            "latent": z,
            "edge_index": edge_index,
            "edge_probs": edge_probs,
        }
