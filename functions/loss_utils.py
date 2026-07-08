from __future__ import annotations

import torch


def normalized_edge_length_loss(edge_probs: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    """Topology loss: penalize probable long edges using normalized distances.

    Requires node coordinates. If coordinates are not available, the caller
    should disable this term or pass lambda_topology=0.
    """
    if pos is None:
        raise ValueError("normalized_edge_length_loss requires node coordinates in `pos`.")

    distances = torch.cdist(pos, pos, p=2)
    distances = distances / distances.max().clamp_min(1e-8)

    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    weighted_lengths = edge_probs[mask] * distances[mask]
    normalizer = edge_probs[mask].sum().clamp_min(1e-8)
    return weighted_lengths.sum() / normalizer


def combine_task_and_topology_loss(
    task_loss: torch.Tensor,
    edge_probs: torch.Tensor,
    pos: torch.Tensor | None,
    alpha_task: float = 1.0,
    use_topology_loss: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute alpha * task + (1-alpha) * topology.

    With use_topology_loss=False, the model trains only on the task loss.
    """
    alpha = float(alpha_task)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_task must be between 0 and 1.")

    zero = task_loss.new_tensor(0.0)
    if use_topology_loss:
        topology_loss = normalized_edge_length_loss(edge_probs=edge_probs, pos=pos)
        total_loss = alpha * task_loss + (1.0 - alpha) * topology_loss
    else:
        topology_loss = zero
        total_loss = task_loss

    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "topology_loss": topology_loss,
    }
