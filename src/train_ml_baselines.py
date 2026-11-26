from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/processed/ml/spatial_demand_proxy_dataset.csv")
CV_PATH = Path("data/processed/ml/spatial_cv_assignments.csv")

OUTPUT_DIR = Path("outputs/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading ML dataset...")
df = pd.read_csv(DATA_PATH)

print("Loading spatial CV assignments...")
cv = pd.read_csv(CV_PATH)

df = df.merge(
    cv[["node_id", "cv_fold"]],
    on="node_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == 5453
assert df["cv_fold"].notna().all()
assert df["cv_fold"].nunique() == 5

TARGET = "demand_proxy_count"
FEATURES = [
    c for c in df.columns
    if c not in ["node_id", TARGET, "cv_fold"]
]

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

print("Rows:", len(df))
print("Predictors:", len(FEATURES))
print("Spatial folds:", df["cv_fold"].nunique())

model_names = [
    "Zero Baseline",
    "Training Mean Baseline",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoosting"
]

oof_predictions = {
    name: np.full(len(df), np.nan)
    for name in model_names
}

fold_results = []
start_time = time.time()

for fold in sorted(df["cv_fold"].unique()):
    train_idx = df.index[df["cv_fold"] != fold]
    test_idx = df.index[df["cv_fold"] == fold]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    print(f"\nTraining fold {fold}...")
    print("Training rows:", len(train_idx))
    print("Validation rows:", len(test_idx))

    predictions = {}

    predictions["Zero Baseline"] = np.zeros(len(test_idx))

    predictions["Training Mean Baseline"] = np.full(
        len(test_idx),
        y_train.mean()
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        )
    }

    for name, model in models.items():
        print("Fitting:", name)

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        predictions[name] = np.clip(pred, 0, None)

    for name, pred in predictions.items():
        oof_predictions[name][test_idx] = pred

        fold_results.append({
            "fold": int(fold),
            "model": name,
            "validation_rows": len(test_idx),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })

    print("Fold", fold, "completed.")

print("\nCalculating overall out-of-fold metrics...")

summary = []

for name in model_names:
    pred = oof_predictions[name]

    assert np.isfinite(pred).all()

    summary.append({
        "model": name,
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    })

summary_df = pd.DataFrame(summary).sort_values("MAE")

fold_df = pd.DataFrame(fold_results)

predictions_df = pd.DataFrame({
    "node_id": df["node_id"],
    "cv_fold": df["cv_fold"],
    "actual_demand_proxy_count": y
})

for name in model_names:
    predictions_df[name] = oof_predictions[name]

summary_path = OUTPUT_DIR / "baseline_model_comparison.csv"
fold_path = OUTPUT_DIR / "baseline_fold_metrics.csv"
pred_path = OUTPUT_DIR / "baseline_oof_predictions.csv"

summary_df.to_csv(summary_path, index=False)
fold_df.to_csv(fold_path, index=False)
predictions_df.to_csv(pred_path, index=False)

report = {
    "experiment": "Spatial ML baseline regression",
    "target": TARGET,
    "target_interpretation": "OSM-based urban-activity demand proxy, not measured EV demand",
    "rows": len(df),
    "predictor_count": len(FEATURES),
    "spatial_folds": int(df["cv_fold"].nunique()),
    "validation_method": "5-fold GroupKFold using 500m spatial blocks",
    "models": summary_df.to_dict(orient="records"),
    "runtime_seconds": round(time.time() - start_time, 2)
}

report_path = OUTPUT_DIR / "baseline_experiment_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nBASELINE MODEL COMPARISON")
print(summary_df.to_string(index=False))

print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
print("\nSaved outputs:")
print(summary_path)
print(fold_path)
print(pred_path)
print(report_path)

print("\nBASELINE EXPERIMENT COMPLETED")
