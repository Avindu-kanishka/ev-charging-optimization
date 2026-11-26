import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASELINE_PATH = "outputs/ml/baseline_oof_predictions.csv"
GNN_PATH = "outputs/ml/gnn/graphsage_oof_predictions.csv"
OUTPUT_DIR = Path("outputs/ml/gnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

baseline = pd.read_csv(BASELINE_PATH)
gnn = pd.read_csv(GNN_PATH)

# Validate unique node identifiers.
assert baseline["node_id"].is_unique
assert gnn["node_id"].is_unique

# Align all predictions by node ID.
baseline = baseline.set_index("node_id").sort_index()
gnn = gnn.set_index("node_id").sort_index()

if not baseline.index.equals(gnn.index):
    raise ValueError("Node IDs do not match between prediction files.")

if not np.array_equal(
    baseline["cv_fold"].to_numpy(),
    gnn["cv_fold"].to_numpy(),
):
    raise ValueError("Spatial CV assignments do not match.")

if not np.allclose(
    baseline["actual_demand_proxy_count"].to_numpy(),
    gnn["actual_demand_proxy_count"].to_numpy(),
):
    raise ValueError("Actual target values do not match.")

actual_count = baseline["actual_demand_proxy_count"].to_numpy()
y_true = (actual_count > 0).astype(int)

model_scores = {
    "Random Forest": baseline["Random Forest"].to_numpy(),
    "Extra Trees": baseline["Extra Trees"].to_numpy(),
    "HistGradientBoosting": baseline[
        "HistGradientBoosting"
    ].to_numpy(),
    "GraphSAGE": gnn["predicted_probability"].to_numpy(),
}

fold_assignments = baseline["cv_fold"].to_numpy()

def evaluate(y, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Invalid prediction scores detected.")

    predicted = (scores >= THRESHOLD).astype(int)

    return {
        "precision": float(
            precision_score(y, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predicted, zero_division=0)
        ),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "predicted_positive_nodes": int(predicted.sum()),
        "actual_positive_nodes": int(y.sum()),
    }

overall_rows = []
fold_rows = []

for model_name, scores in model_scores.items():
    overall = evaluate(y_true, scores)

    overall_rows.append({
        "model": model_name,
        **overall,
    })

    for fold in sorted(np.unique(fold_assignments)):
        mask = fold_assignments == fold

        fold_metrics = evaluate(
            y_true[mask],
            scores[mask],
        )

        fold_rows.append({
            "model": model_name,
            "fold": int(fold),
            **fold_metrics,
        })

overall_df = pd.DataFrame(overall_rows)
fold_df = pd.DataFrame(fold_rows)

overall_df = overall_df.sort_values(
    "f1", ascending=False
).reset_index(drop=True)

overall_path = OUTPUT_DIR / "model_comparison_fixed_threshold.csv"
fold_path = OUTPUT_DIR / "model_comparison_fold_metrics.csv"

overall_df.to_csv(overall_path, index=False)
fold_df.to_csv(fold_path, index=False)

# Build a report with the experimental conditions.
report = {
    "experiment": "GraphSAGE versus classical ML comparison",
    "task": "Binary screening of positive OSM demand-proxy nodes",
    "positive_target_definition": "demand_proxy_count > 0",
    "classification_threshold": THRESHOLD,
    "evaluation": "Same five spatial folds and same 5453 nodes",
    "models": list(model_scores.keys()),
    "score_interpretation": {
        "Random Forest": "Predicted proxy count",
        "Extra Trees": "Predicted proxy count",
        "HistGradientBoosting": "Predicted proxy count",
        "GraphSAGE": "Predicted probability of positive proxy node",
    },
    "overall_results": overall_df.to_dict(orient="records"),
    "fold_mean_f1": {
        name: float(group["f1"].mean())
        for name, group in fold_df.groupby("model")
    },
    "limitations": [
        "Classical ML models produce count estimates, while GraphSAGE produces probabilities.",
        "A common 0.50 threshold enables a consistent screening comparison but does not guarantee optimal calibration.",
        "The target represents OSM activity proxies, not observed EV charging demand.",
        "This is a fixed-threshold comparison, not nested threshold optimization.",
    ],
}

report_path = OUTPUT_DIR / "model_comparison_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n=== OVERALL MODEL COMPARISON ===")
print(
    overall_df[
        [
            "model",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "predicted_positive_nodes",
        ]
    ].to_string(index=False)
)

print("\n=== MEAN F1 BY FIVE SPATIAL FOLDS ===")
print(
    fold_df.groupby("model")["f1"]
    .mean()
    .sort_values(ascending=False)
    .to_string()
)

print("\nSaved:", overall_path)
print("Saved:", fold_path)
print("Saved:", report_path)

print("\nMODEL COMPARISON COMPLETED")
