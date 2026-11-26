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
OUTPUT_DIR = Path("outputs/ml/gnn/nested_threshold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HIDDEN_DIM = 32
DROPOUT = 0.30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
EPOCHS = 120
FIXED_THRESHOLD = 0.50
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.025), 3)

DEVICE = torch.device("cpu")
torch.set_num_threads(2)


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
                0, target,
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
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = GraphSAGELayer(input_dim, HIDDEN_DIM)
        self.layer2 = GraphSAGELayer(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.layer2(x, edge_index))
        return self.output(x).squeeze(1)


def make_subgraph(edge_index, mask):
    indices = np.flatnonzero(mask)

    mapping = np.full(len(mask), -1, dtype=np.int64)
    mapping[indices] = np.arange(len(indices))

    source, target = edge_index
    keep = mask[source] & mask[target]

    local_edges = np.vstack([
        mapping[source[keep]],
        mapping[target[keep]],
    ])

    return (
        indices,
        torch.tensor(local_edges, dtype=torch.long, device=DEVICE),
    )


def train_and_predict(
    X, y, edge_index, train_mask, predict_mask, seed
):
    train_indices, train_edges = make_subgraph(
        edge_index, train_mask
    )
    predict_indices, predict_edges = make_subgraph(
        edge_index, predict_mask
    )

    X_train = X[train_indices]
    X_predict = X[predict_indices]

    y_train = y[train_indices]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_predict = (X_predict - mean) / std

    x_train = torch.tensor(
        X_train, dtype=torch.float32, device=DEVICE
    )
    x_predict = torch.tensor(
        X_predict, dtype=torch.float32, device=DEVICE
    )
    y_tensor = torch.tensor(
        y_train, dtype=torch.float32, device=DEVICE
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GraphSAGE(X.shape[1]).to(DEVICE)

    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)

    if positive == 0:
        raise ValueError("Training subset contains no positive nodes.")

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=DEVICE,
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

        logits = model(x_train, train_edges)
        loss = criterion(logits, y_tensor)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(x_predict, predict_edges)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    return predict_indices, probabilities, float(loss.item())


def calculate_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)

    return {
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "predicted_positive_nodes": int(predicted.sum()),
        "actual_positive_nodes": int(y_true.sum()),
    }


def select_threshold(y_true, probabilities):
    results = []

    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": float(threshold),
            "f1": float(
                f1_score(y_true, predicted, zero_division=0)
            ),
            "precision": float(
                precision_score(
                    y_true, predicted, zero_division=0
                )
            ),
            "recall": float(
                recall_score(y_true, predicted, zero_division=0)
            ),
        })

    # Highest F1; on a tie, choose the higher threshold.
    results.sort(
        key=lambda item: (item["f1"], item["threshold"]),
        reverse=True,
    )

    return results[0], results


print("Loading GNN dataset...")
data = np.load(DATA_PATH, allow_pickle=False)

X = data["X"].astype(np.float32)
y = (data["y"] > 0).astype(np.float32)
edge_index = data["edge_index"].astype(np.int64)
folds = data["cv_fold"].astype(np.int64)

unique_folds = sorted(np.unique(folds).tolist())

if len(unique_folds) != 5:
    raise ValueError("Expected five outer spatial folds.")

outer_predictions = np.full(len(y), np.nan, dtype=np.float32)
fold_results = []
threshold_results = []
all_inner_results = []

print("Nodes:", len(y))
print("Features:", X.shape[1])
print("Positive nodes:", int(y.sum()))
print("Device:", DEVICE)
print("\nStarting nested spatial threshold evaluation...")

for outer_fold in unique_folds:
    print(f"\n========== OUTER FOLD {outer_fold + 1}/5 ==========")

    outer_train_mask = folds != outer_fold
    outer_test_mask = folds == outer_fold

    inner_predictions = np.full(
        len(y), np.nan, dtype=np.float32
    )

    inner_folds = sorted(
        np.unique(folds[outer_train_mask]).tolist()
    )

    # Inner spatial CV: select threshold using outer training data only.
    for inner_fold in inner_folds:
        inner_train_mask = (
            outer_train_mask & (folds != inner_fold)
        )
        inner_validation_mask = (
            outer_train_mask & (folds == inner_fold)
        )

        indices, probabilities, loss = train_and_predict(
            X, y, edge_index,
            inner_train_mask,
            inner_validation_mask,
            SEED + outer_fold * 100 + inner_fold,
        )

        inner_predictions[indices] = probabilities

        print(
            f"Inner fold {inner_fold + 1}: "
            f"validation nodes={len(indices)}, "
            f"loss={loss:.4f}"
        )

    inner_indices = np.flatnonzero(outer_train_mask)

    if np.isnan(inner_predictions[inner_indices]).any():
        raise ValueError("Missing inner out-of-fold predictions.")

    selected, threshold_table = select_threshold(
        y[inner_indices].astype(int),
        inner_predictions[inner_indices],
    )

    selected_threshold = selected["threshold"]

    threshold_results.append({
        "outer_fold": int(outer_fold),
        **selected,
    })

    for row in threshold_table:
        all_inner_results.append({
            "outer_fold": int(outer_fold),
            **row,
        })

    print(
        f"Selected threshold: {selected_threshold:.3f} "
        f"(inner F1={selected['f1']:.4f})"
    )

    # Retrain on the complete outer training set.
    # Predict the untouched outer test subgraph.
    indices, probabilities, loss = train_and_predict(
        X, y, edge_index,
        outer_train_mask,
        outer_test_mask,
        SEED + 1000 + outer_fold,
    )

    outer_predictions[indices] = probabilities

    y_test = y[indices].astype(int)

    fixed_metrics = calculate_metrics(
        y_test, probabilities, FIXED_THRESHOLD
    )
    tuned_metrics = calculate_metrics(
        y_test, probabilities, selected_threshold
    )

    fold_results.append({
        "outer_fold": int(outer_fold),
        "selected_threshold": selected_threshold,
        "inner_validation_f1": selected["f1"],
        "training_loss": loss,
        "test_nodes": int(len(indices)),
        **{
            f"fixed_{key}": value
            for key, value in fixed_metrics.items()
        },
        **{
            f"tuned_{key}": value
            for key, value in tuned_metrics.items()
        },
    })

    print(
        f"Outer test F1: fixed={fixed_metrics['f1']:.4f}, "
        f"tuned={tuned_metrics['f1']:.4f}"
    )

if np.isnan(outer_predictions).any():
    raise ValueError("Missing outer out-of-fold predictions.")

# Pool predictions from untouched outer folds.
fixed_overall = calculate_metrics(
    y.astype(int), outer_predictions, FIXED_THRESHOLD
)

tuned_predictions = np.zeros(len(y), dtype=int)

for result in fold_results:
    fold = result["outer_fold"]
    mask = folds == fold
    tuned_predictions[mask] = (
        outer_predictions[mask] >= result["selected_threshold"]
    ).astype(int)

# Calculate pooled tuned metrics using fold-specific thresholds.
pooled_tuned = {
    "precision": float(
        precision_score(y, tuned_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y, tuned_predictions, zero_division=0)
    ),
    "f1": float(f1_score(y, tuned_predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y, outer_predictions)),
    "average_precision": float(
        average_precision_score(y, outer_predictions)
    ),
    "predicted_positive_nodes": int(tuned_predictions.sum()),
    "actual_positive_nodes": int(y.sum()),
}

fold_df = pd.DataFrame(fold_results)
threshold_df = pd.DataFrame(threshold_results)
inner_df = pd.DataFrame(all_inner_results)

fold_df.to_csv(
    OUTPUT_DIR / "nested_graphsage_fold_metrics.csv",
    index=False,
)
threshold_df.to_csv(
    OUTPUT_DIR / "nested_graphsage_selected_thresholds.csv",
    index=False,
)
inner_df.to_csv(
    OUTPUT_DIR / "nested_graphsage_inner_threshold_search.csv",
    index=False,
)

prediction_df = pd.DataFrame({
    "node_id": data["node_ids"].astype(str),
    "cv_fold": folds,
    "actual_positive": y.astype(int),
    "predicted_probability": outer_predictions,
    "fixed_threshold_prediction": (
        outer_predictions >= FIXED_THRESHOLD
    ).astype(int),
    "nested_threshold_prediction": tuned_predictions,
    "selected_threshold": [
        threshold_results[unique_folds.index(fold)]["threshold"]
        for fold in folds
    ],
})

prediction_df.to_csv(
    OUTPUT_DIR / "nested_graphsage_oof_predictions.csv",
    index=False,
)

report = {
    "experiment": "Nested spatial threshold optimization for GraphSAGE",
    "task": "Binary OSM demand-proxy presence classification",
    "outer_validation": "Five-fold spatial CV",
    "inner_validation": "Four-fold spatial CV within each outer training set",
    "threshold_candidates": THRESHOLDS.tolist(),
    "threshold_selection": (
        "Maximum inner OOF F1; higher threshold breaks ties"
    ),
    "model": {
        "architecture": "Two-layer GraphSAGE",
        "hidden_dimension": HIDDEN_DIM,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(DEVICE),
    },
    "fixed_threshold_overall_oof": fixed_overall,
    "nested_threshold_overall_oof": pooled_tuned,
    "mean_outer_fold_f1": {
        "fixed": float(fold_df["fixed_f1"].mean()),
        "nested": float(fold_df["tuned_f1"].mean()),
    },
    "selected_thresholds": threshold_results,
    "limitations": [
        "Thresholds are selected using inner spatial validation only.",
        "Outer test folds are not used for threshold selection.",
        "The target is OSM activity proxy presence, not observed EV charging demand.",
        "This experiment does not establish external geographic transfer.",
    ],
}

report_path = OUTPUT_DIR / "nested_graphsage_threshold_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n========== FINAL NESTED RESULTS ==========")
print("Fixed threshold OOF:")
print(json.dumps(fixed_overall, indent=2))
print("\nNested threshold OOF:")
print(json.dumps(pooled_tuned, indent=2))
print("\nMean outer-fold F1:")
print(json.dumps(report["mean_outer_fold_f1"], indent=2))
print("\nSelected thresholds:")
print(threshold_df.to_string(index=False))

print("\nReport saved:", report_path)
print("\nNESTED GRAPHSAGE EXPERIMENT COMPLETED")
