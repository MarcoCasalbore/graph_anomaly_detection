# Graph Anomaly Detection with DGM

Self-contained notebook project for a first node-level graph anomaly detection pipeline inspired by DGM-style graph learning and context-aware graph anomaly detection.

## Structure

```text
graph_anomaly_detection/
    notebook.ipynb
    functions/
        data_utils.py
        graph_utils.py
        dgm_module.py
        loss_utils.py
        plot_utils.py
    data/
    results/
    requirements.txt
```

## Environment

Tested with Python 3.12.3.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If GPU/CUDA support is needed, install PyTorch following the official instructions for the target machine.

## Current Scope

The first notebook uses a synthetic water-like dataset and feature aggregates per node. It supports:

- node-level synthetic anomaly detection;
- DGM-style graph learning from a weak initial backbone;
- task-only or task-plus-topology loss;
- graph history collection;
- consensus graph construction;
- anomaly and topology evaluation;
- simple visualizations.

Temporal-window and edge-level anomaly detection are planned extensions.

At this time the architecture is designed to work only in a supervised setting (for the anomaly detection task), but a semi-supervised setting could be a possible extension. 
