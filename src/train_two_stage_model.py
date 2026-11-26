from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = Path("data/processed/ml/spatial_demand_proxy_dataset.csv")
CV_PATH = Path("data/processed/ml/spatial_cv_assignments.csv")
OUTPUT_DIR = Path("outputs/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")

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
y = df[TARGET].astype(float)
y_binary = (y > 0).astype(int)

model_types = ["Random Forest", "Extra Trees"]

predictions = {
    name: np.full(len(df), np.nan)
    for name in model_types
}

probabilities = {
    name: np.full(len(df), np.nan)
    for name in model_types
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

    y_binary_train = y_binary.loc[train_idx]

    positive_train = y_train > 0

    print(f"\nTraining fold {fold}...")

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    regressors = {
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
        )
    }

    for name in model_types:

        classifier = classifiers[name]
        regressor = regressors[name]

        print("Fitting classifier:", name)

        classifier.fit(X_train, y_binary_train)

        probability = classifier.predict_proba(X_test)[:, 1]

        print("Fitting positive-demand regressor:", name)

        regressor.fit(
            X_train.loc[positive_train],
            y_train.loc[positive_train]
        )

        positive_intensity = np.clip(
            regressor.predict(X_test),
            0,
            None
        )

        expected_count = probability * positive_intensity

        probabilities[name][test_idx] = probability
        predictions[name][test_idx] = expected_count

        fold_results.append({
            "fold": int(fold),
            "model": name,
            "positive_training_nodes": int(positive_train.sum()),
            "validation_nodes": len(test_idx),
            "MAE": mean_absolute_error(y_test, expected_count),
            "RMSE": np.sqrt(mean_squared_error(y_test, expected_count)),
            "R2": r2_score(y_test, expected_count)
        })

        print(name, "completed.")

print("\nEvaluating out-of-fold predictions...")

summary = []

for name in model_types:

    pred = predictions[name]
    prob = probabilities[name]

    assert np.isfinite(pred).all()
    assert np.isfinite(prob).all()

    binary_pred = (prob >= 0.5).astype(int)

    summary.append({
        "model": name,
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
        "Precision": precision_score(y_binary, binary_pred, zero_division=0),
        "Recall": recall_score(y_binary, binary_pred, zero_division=0),
        "F1": f1_score(y_binary, binary_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_binary, prob),
        "Positive_node_MAE": mean_absolute_error(
            y[y > 0],
            pred[y > 0]
        )
    })

summary_df = pd.DataFrame(summary)

fold_df = pd.DataFrame(fold_results)

prediction_df = pd.DataFrame({
    "node_id": df["node_id"],
    "cv_fold": df["cv_fold"],
    "actual_demand_proxy_count": y
})

for name in model_types:
    prediction_df[f"{name}_probability"] = probabilities[name]
    prediction_df[f"{name}_expected_count"] = predictions[name]

summary_df.to_csv(
    OUTPUT_DIR / "two_stage_model_comparison.csv",
    index=False
)

fold_df.to_csv(
    OUTPUT_DIR / "two_stage_fold_metrics.csv",
    index=False
)

prediction_df.to_csv(
    OUTPUT_DIR / "two_stage_oof_predictions.csv",
    index=False
)

report = {
    "experiment": "Two-stage spatial demand proxy model",
    "target": "OSM-based demand proxy count",
    "validation": "5-fold spatial GroupKFold",
    "block_size_m": 500,
    "models": summary_df.to_dict(orient="records"),
    "runtime_seconds": round(time.time() - start_time, 2)
}

with open(
    OUTPUT_DIR / "two_stage_experiment_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)

print("\nTWO-STAGE MODEL RESULTS")
print(summary_df.to_string(index=False))

print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
print("\nTWO-STAGE EXPERIMENT COMPLETED")
