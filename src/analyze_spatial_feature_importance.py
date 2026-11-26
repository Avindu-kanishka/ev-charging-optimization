from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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

FEATURES = [
    c for c in df.columns
    if c not in ["node_id", TARGET, "cv_fold"]
]

X = df[FEATURES].astype(float)
y = (df[TARGET] > 0).astype(int)

models = {
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
start_time = time.time()

for model_name, model_template in models.items():

    print(f"\nMODEL: {model_name}")

    for fold in sorted(df["cv_fold"].unique()):

        train_idx = df.index[df["cv_fold"] != fold]
        test_idx = df.index[df["cv_fold"] == fold]

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]

        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        model = model_template

        print(f"Training fold {fold}...")

        model.fit(X_train, y_train)

        baseline_auc = roc_auc_score(
            y_test,
            model.predict_proba(X_test)[:, 1]
        )

        print("Held-out ROC-AUC:", round(baseline_auc, 4))
        print("Calculating permutation importance...")

        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        for feature, mean, std in zip(
            FEATURES,
            importance.importances_mean,
            importance.importances_std
        ):
            results.append({
                "model": model_name,
                "fold": int(fold),
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
                "baseline_roc_auc": float(baseline_auc)
            })

        print(f"Fold {fold} completed.")

importance_df = pd.DataFrame(results)

importance_df.to_csv(
    OUTPUT_DIR / "spatial_permutation_importance_by_fold.csv",
    index=False
)

summary = (
    importance_df
    .groupby(["model", "feature"])
    .agg(
        mean_importance=("importance_mean", "mean"),
        std_across_folds=("importance_mean", "std"),
        mean_repeat_std=("importance_std", "mean")
    )
    .reset_index()
    .sort_values(
        ["model", "mean_importance"],
        ascending=[True, False]
    )
)

summary.to_csv(
    OUTPUT_DIR / "spatial_permutation_importance_summary.csv",
    index=False
)

print("\nTOP 15 FEATURES")

for model_name in models:
    print(f"\n--- {model_name} ---")
    print(
        summary[summary["model"] == model_name]
        .head(15)
        .round(5)
        .to_string(index=False)
    )

print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
print("\nFEATURE IMPORTANCE EXPERIMENT COMPLETED")
