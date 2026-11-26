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
OUTPUT_DIR = Path("outputs/ml/gnn/architecture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 120
WEIGHT_DECAY = 0.0001
THRESHOLD = 0.50
SEED = 42

DEVICE = torch.device("cpu")
torch.set_num_threads(2)

CONFIGS = {
    "Baseline_2L_32H": {
        "layers": 2, "hidden": 32, "dropout": 0.30, "lr": 0.010,
    },
    "A_1L_32H": {
        "layers": 1, "hidden": 32, "dropout": 0.30, "lr": 0.010,
    },
    "B_2L_64H": {
        "layers": 2, "hidden": 64, "dropout": 0.30, "lr": 0.010,
    },
    "C_2L_Dropout50": {
        "layers": 2, "hidden": 32, "dropout": 0.50, "lr": 0.010,
    },
    "D_2L_LR005": {
        "layers": 2, "hidden": 32, "dropout": 0.30, "lr": 0.005,
    },
}


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
    def __init__(self, input_dim, config):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(config["layers"]):
            layers.append(
                GraphSAGELayer(current_dim, config["hidden"])
            )
            current_dim = config["hidden"]

        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(current_dim, 1)
        self.dropout = config["dropout"]

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(
                x, p=self.dropout, training=self.training
            )

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


def train_fold(X, y, edge_index, train_mask, test_mask, config, seed):
    train_indices, train_edges = make_subgraph(edge_index, train_mask)
    test_indices, test_edges = make_subgraph(edge_index, test_mask)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    x_train = torch.tensor(
        X_train, dtype=torch.float32, device=DEVICE
    )
    x_test = torch.tensor(
        X_test, dtype=torch.float32, device=DEVICE
    )
    y_tensor = torch.tensor(
        y_train, dtype=torch.float32, device=DEVICE
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GraphSAGE(X.shape[1], config).to(DEVICE)

    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)

    if positive == 0:
        raise ValueError("Training subset has no positive nodes.")

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits = model(x_train, train_edges)
        loss = criterion(logits, y_tensor)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(x_test, test_edges)
        ).cpu().numpy()

    return test_indices, probabilities, float(loss.item())


def metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "predicted_positive_nodes": int(predictions.sum()),
    }


print("Loading graph dataset...")
data = np.load(DATA_PATH, allow_pickle=False)

X = data["X"].astype(np.float32)
y = (data["y"] > 0).astype(np.float32)
edge_index = data["edge_index"].astype(np.int64)
folds = data["cv_fold"].astype(np.int64)

all_fold_results = []
all_predictions = []
summary_rows = []

for config_name, config in CONFIGS.items():
    print(f"\n========== {config_name} ==========")
    oof = np.full(len(y), np.nan, dtype=np.float32)

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        test_mask = folds == fold

        indices, probabilities, loss = train_fold(
            X, y, edge_index,
            train_mask, test_mask,
            config,
            SEED + int(fold),
        )

        oof[indices] = probabilities

        fold_metrics = metrics(y[indices].astype(int), probabilities)

        all_fold_results.append({
            "configuration": config_name,
            "fold": int(fold),
            "training_loss": loss,
            "test_nodes": int(len(indices)),
            **fold_metrics,
        })

        print(
            f"Fold {fold + 1}/5: "
            f"F1={fold_metrics['f1']:.4f}, "
            f"AUC={fold_metrics['roc_auc']:.4f}"
        )

    if np.isnan(oof).any():
        raise ValueError(f"Missing predictions in {config_name}")

    overall = metrics(y.astype(int), oof)

    summary_rows.append({
        "configuration": config_name,
        **config,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        **overall,
    })

    for i in range(len(y)):
        all_predictions.append({
            "configuration": config_name,
            "node_id": str(data["node_ids"][i]),
            "cv_fold": int(folds[i]),
            "actual_positive": int(y[i]),
            "predicted_probability": float(oof[i]),
            "predicted_positive": int(oof[i] >= THRESHOLD),
        })

    print(
        f"Overall: F1={overall['f1']:.4f}, "
        f"AUC={overall['roc_auc']:.4f}, "
        f"AP={overall['average_precision']:.4f}"
    )

summary_df = pd.DataFrame(summary_rows).sort_values(
    "f1", ascending=False
)
fold_df = pd.DataFrame(all_fold_results)
prediction_df = pd.DataFrame(all_predictions)

summary_df.to_csv(
    OUTPUT_DIR / "graphsage_architecture_summary.csv",
    index=False,
)
fold_df.to_csv(
    OUTPUT_DIR / "graphsage_architecture_fold_metrics.csv",
    index=False,
)
prediction_df.to_csv(
    OUTPUT_DIR / "graphsage_architecture_oof_predictions.csv",
    index=False,
)

report = {
    "experiment": "GraphSAGE architecture ablation",
    "evaluation": "Five-fold spatial cross-validation",
    "threshold": THRESHOLD,
    "epochs": EPOCHS,
    "device": str(DEVICE),
    "configurations": CONFIGS,
    "results": summary_df.to_dict(orient="records"),
    "best_configuration_by_f1": str(
        summary_df.iloc[0]["configuration"]
    ),
    "limitations": [
        "Fixed 0.50 classification threshold.",
        "One experiment per configuration with fixed random seeds.",
        "No hyperparameter search beyond the listed configurations.",
        "Target represents OSM activity proxy presence, not observed EV charging demand.",
    ],
}

(OUTPUT_DIR / "graphsage_architecture_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n========== ARCHITECTURE COMPARISON ==========")
print(
    summary_df[
        [
            "configuration",
            "layers",
            "hidden",
            "dropout",
            "lr",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    ].to_string(index=False)
)

print("\nBest configuration:", report["best_configuration_by_f1"])
print("\nGRAPHSAGE ARCHITECTURE EXPERIMENT COMPLETED")
