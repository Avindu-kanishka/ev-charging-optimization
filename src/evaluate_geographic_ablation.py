from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = Path("data/processed/ml/spatial_demand_proxy_dataset.csv")
CV_PATH = Path("data/processed/ml/spatial_cv_assignments.csv")
OUTPUT_DIR = Path("outputs/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
cv = pd.read_csv(CV_PATH)

df = df.merge(
    cv[["node_id", "cv_fold"]],
    on="node_id",
    how="left",
    validate="one_to_one"
)

TARGET = "demand_proxy_count"

all_features = [
    c for c in df.columns
    if c not in ["node_id", TARGET, "cv_fold"]
]

road_features = [
    c for c in all_features
    if c not in ["latitude", "longitude"]
]

feature_sets = {
    "Full Features": all_features,
    "Road Network Only": road_features
}

y = (df[TARGET] > 0).astype(int)

results = []
predictions = {}
start_time = time.time()

for experiment, features in feature_sets.items():

    print(f"\nEXPERIMENT: {experiment}")
    print("Predictor count:", len(features))

    X = df[features].astype(float)
    predictions[experiment] = np.full(len(df), np.nan)

    for fold in sorted(df["cv_fold"].unique()):

        train_idx = df.index[df["cv_fold"] != fold]
        test_idx = df.index[df["cv_fold"] == fold]

        model = ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        print("Training fold:", fold)

        model.fit(X.loc[train_idx], y.loc[train_idx])

        positive_position = list(model.classes_).index(1)

        probability = model.predict_proba(
            X.loc[test_idx]
        )[:, positive_position]

        auc = roc_auc_score(y.loc[test_idx], probability)

        predictions[experiment][test_idx] = probability

        results.append({
            "experiment": experiment,
            "fold": int(fold),
            "predictor_count": len(features),
            "roc_auc": auc
        })

        print("ROC-AUC:", round(auc, 4))

fold_df = pd.DataFrame(results)

summary = (
    fold_df.groupby("experiment")
    .agg(
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        minimum_roc_auc=("roc_auc", "min"),
        maximum_roc_auc=("roc_auc", "max"),
        predictor_count=("predictor_count", "first")
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "geographic_ablation_summary.csv",
    index=False
)

fold_df.to_csv(
    OUTPUT_DIR / "geographic_ablation_fold_metrics.csv",
    index=False
)

prediction_df = pd.DataFrame({
    "node_id": df["node_id"],
    "cv_fold": df["cv_fold"],
    "actual_positive": y
})

for experiment in feature_sets:
    prediction_df[
        experiment.replace(" ", "_").lower() + "_probability"
    ] = predictions[experiment]

prediction_df.to_csv(
    OUTPUT_DIR / "geographic_ablation_oof_predictions.csv",
    index=False
)

print("\nGEOGRAPHIC ABLATION SUMMARY")
print(summary.round(4).to_string(index=False))

print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
print("\nGEOGRAPHIC ABLATION COMPLETED")
