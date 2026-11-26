from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = Path("data/processed/ml/spatial_demand_proxy_dataset.csv")
CV_PATH = Path("data/processed/ml/spatial_cv_assignments.csv")
OUTPUT_DIR = Path("outputs/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
cv = pd.read_csv(CV_PATH)

df = df.merge(
    cv[["node_id", "cv_fold", "spatial_block"]],
    on="node_id",
    how="left",
    validate="one_to_one"
)

TARGET = "demand_proxy_count"
FEATURES = [
    c for c in df.columns
    if c not in ["node_id", TARGET, "cv_fold", "spatial_block"]
]

X = df[FEATURES].astype(float)
y = (df[TARGET] > 0).astype(int)
groups = df["spatial_block"]

def create_classifier(name):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

    return ExtraTreesClassifier(
        n_estimators=200,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

models = ["Random Forest", "Extra Trees"]
thresholds = np.arange(0.10, 0.901, 0.025)

outer_predictions = {
    name: np.full(len(df), np.nan)
    for name in models
}

outer_thresholds = []
fold_results = []

start_time = time.time()

print("STARTING NESTED SPATIAL VALIDATION")
print("Outer folds:", df["cv_fold"].nunique())
print("Inner folds: 4")
print("Threshold range: 0.10 to 0.90")

for outer_fold in sorted(df["cv_fold"].unique()):

    outer_train_idx = df.index[df["cv_fold"] != outer_fold]
    outer_test_idx = df.index[df["cv_fold"] == outer_fold]

    X_outer_train = X.loc[outer_train_idx]
    y_outer_train = y.loc[outer_train_idx]
    groups_outer_train = groups.loc[outer_train_idx]

    X_outer_test = X.loc[outer_test_idx]
    y_outer_test = y.loc[outer_test_idx]

    print(f"\nOUTER FOLD {outer_fold}")

    inner_splitter = GroupKFold(n_splits=4)

    for name in models:

        print("Model:", name)
        inner_probabilities = np.full(len(outer_train_idx), np.nan)

        for inner_train_pos, inner_valid_pos in inner_splitter.split(
            X_outer_train,
            y_outer_train,
            groups=groups_outer_train
        ):

            X_inner_train = X_outer_train.iloc[inner_train_pos]
            y_inner_train = y_outer_train.iloc[inner_train_pos]

            X_inner_valid = X_outer_train.iloc[inner_valid_pos]

            classifier = create_classifier(name)

            classifier.fit(X_inner_train, y_inner_train)

            positive_class_position = list(
                classifier.classes_
            ).index(1)

            inner_probabilities[inner_valid_pos] = (
                classifier.predict_proba(X_inner_valid)[
                    :, positive_class_position
                ]
            )

        assert np.isfinite(inner_probabilities).all()

        best_threshold = None
        best_f1 = -1

        for threshold in thresholds:

            inner_predictions = (
                inner_probabilities >= threshold
            ).astype(int)

            score = f1_score(
                y_outer_train,
                inner_predictions,
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        print(
            "Selected threshold:", round(best_threshold, 3),
            "| Inner F1:", round(best_f1, 4)
        )

        final_classifier = create_classifier(name)

        final_classifier.fit(X_outer_train, y_outer_train)

        positive_class_position = list(
            final_classifier.classes_
        ).index(1)

        test_probabilities = final_classifier.predict_proba(
            X_outer_test
        )[:, positive_class_position]

        test_predictions = (
            test_probabilities >= best_threshold
        ).astype(int)

        outer_predictions[name][outer_test_idx] = test_probabilities

        precision = precision_score(
            y_outer_test,
            test_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_outer_test,
            test_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_outer_test,
            test_predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_outer_test,
            test_probabilities
        )

        fold_results.append({
            "outer_fold": int(outer_fold),
            "model": name,
            "selected_threshold": best_threshold,
            "inner_f1": best_f1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "predicted_positive": int(test_predictions.sum()),
            "actual_positive": int(y_outer_test.sum())
        })

        outer_thresholds.append({
            "outer_fold": int(outer_fold),
            "model": name,
            "selected_threshold": best_threshold
        })

        print(
            "Outer results:",
            "Precision =", round(precision, 4),
            "Recall =", round(recall, 4),
            "F1 =", round(f1, 4),
            "ROC-AUC =", round(auc, 4)
        )

print("\nCALCULATING FINAL NESTED VALIDATION RESULTS")

fold_df = pd.DataFrame(fold_results)
summary = []

for name in models:

    probability = outer_predictions[name]
    assert np.isfinite(probability).all()

    model_folds = fold_df[fold_df["model"] == name]

    summary.append({
        "model": name,
        "mean_precision": model_folds["precision"].mean(),
        "mean_recall": model_folds["recall"].mean(),
        "mean_f1": model_folds["f1"].mean(),
        "std_f1": model_folds["f1"].std(),
        "mean_roc_auc": model_folds["roc_auc"].mean(),
        "pooled_roc_auc": roc_auc_score(y, probability),
        "mean_selected_threshold": model_folds[
            "selected_threshold"
        ].mean()
    })

summary_df = pd.DataFrame(summary)

predictions_df = pd.DataFrame({
    "node_id": df["node_id"],
    "outer_fold": df["cv_fold"],
    "actual_positive": y
})

for name in models:
    predictions_df[name + "_probability"] = outer_predictions[name]

fold_df.to_csv(
    OUTPUT_DIR / "nested_spatial_fold_metrics.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "nested_spatial_model_comparison.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR / "nested_spatial_oof_predictions.csv",
    index=False
)

report = {
    "experiment": "Nested spatial threshold validation",
    "outer_validation": "5-fold spatial GroupKFold",
    "inner_validation": "4-fold spatial GroupKFold",
    "threshold_selection": "Maximum inner-validation F1",
    "thresholds_tested": [round(float(t), 3) for t in thresholds],
    "target": "OSM-based demand proxy presence",
    "results": summary_df.to_dict(orient="records"),
    "runtime_seconds": round(time.time() - start_time, 2)
}

with open(
    OUTPUT_DIR / "nested_spatial_validation_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)

print("\nFINAL NESTED SPATIAL RESULTS")
print(summary_df.round(4).to_string(index=False))

print("\nFOLD-BY-FOLD RESULTS")
print(fold_df.round(4).to_string(index=False))

print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
print("\nNESTED SPATIAL VALIDATION COMPLETED")
