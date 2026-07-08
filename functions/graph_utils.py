from __future__ import annotations

import numpy as np


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    """Return an undirected edge with stable node order."""
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def edge_index_to_edge_set(edge_index: np.ndarray, undirected: bool = True) -> set[tuple[int, int]]:
    """Convert edge_index with shape [2, E] to a Python edge set."""
    if edge_index is None or edge_index.size == 0:
        return set()
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    edges = set()
    for u, v in zip(src, dst):
        if u == v:
            continue
        edges.add(canonical_edge(u, v) if undirected else (int(u), int(v)))
    return edges


def edge_set_to_edge_index(edges: set[tuple[int, int]], bidirectional: bool = True) -> np.ndarray:
    """Convert an edge set to edge_index with shape [2, E]."""
    rows = []
    for u, v in sorted(edges):
        rows.append((int(u), int(v)))
        if bidirectional:
            rows.append((int(v), int(u)))
    if not rows:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64).T


def edge_index_to_adjacency(edge_index: np.ndarray, num_nodes: int, undirected: bool = True) -> np.ndarray:
    """Build a binary adjacency matrix from edge_index."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edge_set_to_edge_index(edge_index_to_edge_set(edge_index, undirected=False), False).T:
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            A[u, v] = 1.0
            if undirected:
                A[v, u] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def adjacency_to_edge_index(A: np.ndarray, threshold: float = 0.5, bidirectional: bool = True) -> np.ndarray:
    """Threshold an adjacency/score matrix and return edge_index."""
    A = np.asarray(A)
    n = A.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] >= threshold or A[j, i] >= threshold:
                edges.add((i, j))
    return edge_set_to_edge_index(edges, bidirectional=bidirectional)


def knn_graph_from_positions(pos: np.ndarray, k: int = 3, bidirectional: bool = True) -> np.ndarray:
    """Build a geometric kNN graph from node positions."""
    xy = np.asarray(pos, dtype=np.float32)[:, :2]
    n = xy.shape[0]
    k = min(max(1, int(k)), n - 1)
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            edges.add(canonical_edge(i, int(j)))
    return edge_set_to_edge_index(edges, bidirectional=bidirectional)


def topology_metrics(true_edge_index: np.ndarray, pred_edge_index: np.ndarray, num_nodes: int) -> dict:
    """Undirected edge precision/recall/F1 for topology reconstruction."""
    true_edges = edge_index_to_edge_set(true_edge_index, undirected=True)
    pred_edges = edge_index_to_edge_set(pred_edge_index, undirected=True)
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    possible = num_nodes * (num_nodes - 1) // 2
    shd = fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "structural_hamming_distance": shd,
        "possible_undirected_edges": possible,
    }

