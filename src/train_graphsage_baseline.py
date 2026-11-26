import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = "data/processed/ml/gnn/colombo_gnn_dataset.npz"
OUTPUT_DIR = Path("outputs/ml/gnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HIDDEN_DIM = 32
DROPOUT = 0.30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
EPOCHS = 120
THRESHOLD = 0.50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

device = torch.device("cpu")


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        source, target = edge_index

        neighbor_sum = torch.zeros_like(x)
        degree = torch.zeros(
            (x.shape[0], 1), dtype=x.dtype, device=x.device
        )

        if source.numel() > 0:
            neighbor_sum.index_add_(0, target, x[source])
            degree.index_add_(
                0,
                target,
                torch.ones(
                    (target.shape[0], 1),
                    dtype=x.dtype,
                    device=x.device,
                ),
            )

        neighbor_mean = neighbor_sum / degree.clamp(min=1.0)

        return (
            self.self_linear(x)
            + self.neighbor_linear(neighbor_mean)
        )


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer2(x, edge_index)
        x = F.relu(x)

        return self.output(x).squeeze(1)


def induced_subgraph(edge_index, selected_mask):
    selected_indices = np.flatnonzero(selected_mask)

    global_to_local = np.full(
        len(selected_mask), -1, dtype=np.int64
    )
    global_to_local[selected_indices] = np.arange(
        len(selected_indices)
    )

    source = edge_index[0]
    target = edge_index[1]

    keep = selected_mask[source] & selected_mask[target]

    local_edges = np.vstack([
        global_to_local[source[keep]],
        global_to_local[target[keep]],
    ])

    return (
        selected_indices,
        torch.tensor(local_edges, dtype=torch.long, device=device),
    )


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "predicted_positive_nodes": int(predictions.sum()),
        "actual_positive_nodes": int(y_true.sum()),
    }


print("Loading prepared graph dataset...")
data = np.load(DATA_PATH, allow_pickle=False)

X = data["X"].astype(np.float32)
y_count = data["y"].astype(np.float32)
edge_index = data["edge_index"].astype(np.int64)
folds = data["cv_fold"].astype(np.int64)

y = (y_count > 0).astype(np.float32)

num_nodes, num_features = X.shape
unique_folds = sorted(np.unique(folds).tolist())

if len(unique_folds) != 5:
    raise ValueError("Expected exactly five spatial folds.")

if not np.isfinite(X).all():
    raise ValueError("Input features contain invalid values.")

if not np.isfinite(y).all():
    raise ValueError("Target contains invalid values.")

print("Nodes:", num_nodes)
print("Features:", num_features)
print("Adjacency edges:", edge_index.shape[1])
print("Positive target nodes:", int(y.sum()))
print("Device:", device)
print("\nStarting five-fold spatial GraphSAGE evaluation...")

oof_probabilities = np.full(num_nodes, np.nan, dtype=np.float32)
fold_results = []

for fold in unique_folds:
    print(f"\n--- Fold {fold + 1}/5 ---")

    train_mask = folds != fold
    test_mask = folds == fold

    train_indices, train_edges = induced_subgraph(
        edge_index, train_mask
    )
    test_indices, test_edges = induced_subgraph(
        edge_index, test_mask
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            f"Fold {fold} does not contain both target classes."
        )

    # Fit normalization only on training nodes.
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    X_train_scaled = (X_train - feature_mean) / feature_std
    X_test_scaled = (X_test - feature_mean) / feature_std

    x_train_tensor = torch.tensor(
        X_train_scaled, dtype=torch.float32, device=device
    )
    x_test_tensor = torch.tensor(
        X_test_scaled, dtype=torch.float32, device=device
    )
    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32, device=device
    )

    torch.manual_seed(SEED + fold)

    model = GraphSAGE(
        input_dim=num_features,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits = model(x_train_tensor, train_edges)
        loss = criterion(logits, y_train_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} "
                f"Training loss: {loss.item():.5f}"
            )

    # Predict on the isolated held-out spatial subgraph.
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_tensor, test_edges)
        probabilities = torch.sigmoid(test_logits).cpu().numpy()

    oof_probabilities[test_indices] = probabilities

    metrics = calculate_metrics(y_test.astype(int), probabilities)
    metrics["fold"] = int(fold)
    metrics["train_nodes"] = int(len(train_indices))
    metrics["test_nodes"] = int(len(test_indices))
    metrics["train_edges"] = int(train_edges.shape[1])
    metrics["test_edges"] = int(test_edges.shape[1])
    metrics["training_loss"] = float(loss.item())

    fold_results.append(metrics)

    print(
        f"Fold {fold + 1} results: "
        f"F1={metrics['f1']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"Average Precision={metrics['average_precision']:.4f}"
    )

if np.isnan(oof_probabilities).any():
    raise ValueError("Some nodes do not have out-of-fold predictions.")

y_binary = y.astype(int)
overall_metrics = calculate_metrics(
    y_binary, oof_probabilities
)

fold_df = pd.DataFrame(fold_results)
fold_path = OUTPUT_DIR / "graphsage_fold_metrics.csv"
fold_df.to_csv(fold_path, index=False)

prediction_df = pd.DataFrame({
    "node_index": np.arange(num_nodes),
    "node_id": data["node_ids"].astype(str),
    "cv_fold": folds,
    "actual_demand_proxy_count": y_count,
    "actual_positive": y_binary,
    "predicted_probability": oof_probabilities,
    "predicted_positive": (
        oof_probabilities >= THRESHOLD
    ).astype(int),
})

prediction_path = OUTPUT_DIR / "graphsage_oof_predictions.csv"
prediction_df.to_csv(prediction_path, index=False)

report = {
    "model": "Two-layer GraphSAGE implemented in plain PyTorch",
    "task": "Binary classification: demand_proxy_count > 0",
    "device": str(device),
    "seed": SEED,
    "architecture": {
        "input_features": int(num_features),
        "hidden_dimension": HIDDEN_DIM,
        "layers": 2,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    },
    "evaluation": {
        "method": "Five-fold spatial cross-validation",
        "graph_split": (
            "Training and held-out test nodes use separate "
            "induced subgraphs; cross-fold edges are excluded."
        ),
        "feature_scaling": "Fit on training nodes only",
        "classification_threshold": THRESHOLD,
    },
    "overall_oof_metrics": overall_metrics,
    "mean_fold_metrics": {
        metric: float(fold_df[metric].mean())
        for metric in [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    },
    "fold_metric_standard_deviation": {
        metric: float(fold_df[metric].std(ddof=1))
        for metric in [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    },
    "artifacts": [
        str(fold_path),
        str(prediction_path),
    ],
    "research_limitations": [
        "The target represents OSM activity proxies, not observed EV charging demand.",
        "The model is an initial baseline with a fixed 0.50 threshold.",
        "Hyperparameter tuning and threshold selection have not yet been performed.",
    ],
}

report_path = OUTPUT_DIR / "graphsage_baseline_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n=== GRAPH SAGE OVERALL OOF RESULTS ===")
print(json.dumps(overall_metrics, indent=2))

print("\n=== MEAN FOLD METRICS ===")
print(json.dumps(report["mean_fold_metrics"], indent=2))

print("\nSaved:", fold_path)
print("Saved:", prediction_path)
print("Saved:", report_path)
print("\nGRAPHSAGE BASELINE EXPERIMENT COMPLETED")
