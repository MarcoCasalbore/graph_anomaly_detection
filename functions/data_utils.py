from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph_utils import edge_set_to_edge_index, knn_graph_from_positions


@dataclass
class GraphAnomalyData:
    """Standard container used by the notebook."""

    X: np.ndarray
    y: np.ndarray
    pos: np.ndarray
    edge_index_true: np.ndarray
    edge_index_initial: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    feature_names: list[str]
    metadata: dict


def _make_masks(num_nodes: int, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_nodes)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[order[:n_train]] = True
    val_mask[order[n_train : n_train + n_val]] = True
    test_mask[order[n_train + n_val :]] = True
    return train_mask, val_mask, test_mask


def generate_synthetic_water_like(
    num_nodes: int = 50,
    timesteps: int = 240,
    seed: int = 42,
    anomaly_fraction: float = 0.15,
    anomaly_feature: str = "demand",
) -> GraphAnomalyData:
    """Create a small water-like graph with node-level synthetic anomalies."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(num_nodes, 2)).astype(np.float32)
    edge_index_true = knn_graph_from_positions(pos, k=3, bidirectional=True)

    t = np.arange(timesteps, dtype=np.float32)
    daily = np.sin(2.0 * np.pi * t / 24.0)[:, None]
    node_bias = rng.normal(0.0, 0.2, size=(1, num_nodes))
    demand = 1.0 + 0.25 * daily + node_bias + rng.normal(0.0, 0.05, size=(timesteps, num_nodes))
    pressure = 3.0 - 0.6 * demand + rng.normal(0.0, 0.05, size=(timesteps, num_nodes))

    y = np.zeros(num_nodes, dtype=np.int64)
    n_anom = max(1, int(anomaly_fraction * num_nodes))
    anomaly_nodes = rng.choice(num_nodes, size=n_anom, replace=False)
    y[anomaly_nodes] = 1
    if anomaly_feature in {"demand", "both"}:
        demand[:, anomaly_nodes] += rng.normal(0.8, 0.15, size=(timesteps, n_anom))
    if anomaly_feature in {"pressure", "both"}:
        pressure[:, anomaly_nodes] -= rng.normal(0.6, 0.12, size=(timesteps, n_anom))

    X, feature_names = make_node_features(pressure, demand, pos)
    edge_index_initial = knn_graph_from_positions(pos, k=2, bidirectional=True)
    train_mask, val_mask, test_mask = _make_masks(num_nodes, seed)

    return GraphAnomalyData(
        X=X,
        y=y,
        pos=pos,
        edge_index_true=edge_index_true,
        edge_index_initial=edge_index_initial,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_names=feature_names,
        metadata={"source": "generated_synthetic_water_like", "anomaly_nodes": anomaly_nodes.tolist()},
    )


def make_node_features(pressure: np.ndarray, demand: np.ndarray, pos: np.ndarray):
    """Build simple node-level features from time series and coordinates."""
    pressure = np.asarray(pressure, dtype=np.float32)
    demand = np.asarray(demand, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    X = np.column_stack(
        [
            pos[:, 0],
            pos[:, 1],
            pressure.mean(axis=0),
            pressure.std(axis=0),
            demand.mean(axis=0),
            demand.std(axis=0),
            pressure[-1] - pressure[0],
            demand[-1] - demand[0],
        ]
    ).astype(np.float32)
    names = [
        "x_coord",
        "y_coord",
        "pressure_mean",
        "pressure_std",
        "demand_mean",
        "demand_std",
        "pressure_trend",
        "demand_trend",
    ]
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X, names


def load_synthetic_water_npz(path: str | Path, seed: int = 42, anomaly_fraction: float = 0.15) -> GraphAnomalyData:
    """Load a self-contained synthetic water .npz and add node anomaly labels."""
    path = Path(path)
    data = np.load(path)
    pressure = data["pressure"].astype(np.float32)
    demand = data["demand"].astype(np.float32)
    pos = data["node_xy"].astype(np.float32)
    edge_index_true = data["edge_index"].astype(np.int64)
    edge_index_initial = (
        data["initial_edge_index"].astype(np.int64)
        if "initial_edge_index" in data.files
        else knn_graph_from_positions(pos, k=2, bidirectional=True)
    )

    rng = np.random.default_rng(seed)
    num_nodes = pos.shape[0]
    y = np.zeros(num_nodes, dtype=np.int64)
    n_anom = max(1, int(anomaly_fraction * num_nodes))
    anomaly_nodes = rng.choice(num_nodes, size=n_anom, replace=False)
    y[anomaly_nodes] = 1
    demand = demand.copy()
    demand[:, anomaly_nodes] += rng.normal(0.8, 0.15, size=(demand.shape[0], n_anom))

    X, feature_names = make_node_features(pressure, demand, pos)
    train_mask, val_mask, test_mask = _make_masks(num_nodes, seed)
    return GraphAnomalyData(
        X=X,
        y=y,
        pos=pos,
        edge_index_true=edge_index_true,
        edge_index_initial=edge_index_initial,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_names=feature_names,
        metadata={"source": str(path), "anomaly_nodes": anomaly_nodes.tolist()},
    )


def load_or_create_synthetic_water_dataset(
    data_npz: str | Path | None = None,
    seed: int = 42,
    anomaly_fraction: float = 0.15,
) -> GraphAnomalyData:
    """Load a local .npz if provided, otherwise generate a small fallback dataset."""
    if data_npz is not None and Path(data_npz).exists():
        return load_synthetic_water_npz(data_npz, seed=seed, anomaly_fraction=anomaly_fraction)
    return generate_synthetic_water_like(seed=seed, anomaly_fraction=anomaly_fraction)

