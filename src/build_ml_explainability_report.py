from pathlib import Path
import pandas as pd

importance_path = Path(
    "outputs/ml/spatial_permutation_importance_summary.csv"
)
fold_importance_path = Path(
    "outputs/ml/spatial_permutation_importance_by_fold.csv"
)
ablation_path = Path(
    "outputs/ml/geographic_ablation_summary.csv"
)

output_path = Path(
    "outputs/analysis/ml_explainability_report.md"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

importance = pd.read_csv(importance_path)
fold_importance = pd.read_csv(fold_importance_path)
ablation = pd.read_csv(ablation_path)

lines = []

lines.append("# Explainable Machine Learning Research Report")
lines.append("")
lines.append("## 1. Research objective")
lines.append("")
lines.append(
    "This experiment investigates which geographic and road-network "
    "characteristics contribute to predicting OSM-based urban-activity "
    "demand proxy presence in Colombo."
)
lines.append("")
lines.append(
    "The target represents mapped urban-activity proxy locations, "
    "not observed EV charging sessions or electricity consumption."
)
lines.append("")
lines.append("## 2. Permutation feature importance")
lines.append("")
lines.append(
    "Permutation importance was evaluated using ROC-AUC on five "
    "held-out spatial folds. Higher positive values indicate a greater "
    "reduction in predictive performance when a feature is shuffled."
)
lines.append("")

for model in ["Extra Trees", "Random Forest"]:
    model_data = importance[
        importance["model"] == model
    ].sort_values("mean_importance", ascending=False)

    lines.append(f"### 2.{1 if model == 'Extra Trees' else 2} {model}")
    lines.append("")
    lines.append(
        "| Feature | Mean importance | Across-fold SD |"
    )
    lines.append("|---|---:|---:|")

    for _, row in model_data.head(10).iterrows():
        lines.append(
            f"| {row['feature']} | "
            f"{row['mean_importance']:.5f} | "
            f"{row['std_across_folds']:.5f} |"
        )

    lines.append("")

lines.append("## 3. Feature importance stability")
lines.append("")

for model in ["Extra Trees", "Random Forest"]:
    model_data = fold_importance[
        fold_importance["model"] == model
    ]

    positive_counts = (
        model_data.groupby("feature")["importance_mean"]
        .apply(lambda values: int((values > 0).sum()))
        .sort_values(ascending=False)
    )

    lines.append(f"### {model}")
    lines.append("")
    lines.append(
        "| Feature | Folds with positive importance |"
    )
    lines.append("|---|---:|")

    for feature, count in positive_counts.head(10).items():
        lines.append(f"| {feature} | {count}/5 |")

    lines.append("")

lines.append("## 4. Geographic feature ablation")
lines.append("")
lines.append(
    "The Extra Trees classifier was evaluated using the same five "
    "spatial folds with and without latitude and longitude."
)
lines.append("")
lines.append(
    "| Feature configuration | Predictors | Mean ROC-AUC | SD |"
)
lines.append("|---|---:|---:|---:|")

for _, row in ablation.iterrows():
    lines.append(
        f"| {row['experiment']} | "
        f"{int(row['predictor_count'])} | "
        f"{row['mean_roc_auc']:.4f} | "
        f"{row['std_roc_auc']:.4f} |"
    )

full = ablation[
    ablation["experiment"] == "Full Features"
].iloc[0]

road = ablation[
    ablation["experiment"] == "Road Network Only"
].iloc[0]

difference = full["mean_roc_auc"] - road["mean_roc_auc"]

lines.append("")
lines.append(
    f"Removing geographic coordinates reduced mean ROC-AUC by "
    f"{difference:.4f}."
)
lines.append("")
lines.append("## 5. Research interpretation")
lines.append("")
lines.append(
    "- Geographic coordinates are highly influential predictors "
    "in both tree-based models."
)
lines.append(
    "- Road trunk and residential road characteristics also "
    "contribute predictive information."
)
lines.append(
    "- Road-network-only features retain predictive ability, "
    "although performance is lower than the full feature model."
)
lines.append(
    "- Local road-node density contributes less than geographic "
    "position and selected road hierarchy features in these experiments."
)
lines.append("")
lines.append("## 6. Implications for infrastructure planning")
lines.append("")
lines.append(
    "The results suggest that spatial context and road-network "
    "characteristics can jointly support preliminary screening of "
    "potential urban demand areas. Road-network-only predictions may "
    "also be useful where geographic coordinates are unavailable or "
    "where transferable structural patterns are being investigated."
)
lines.append("")
lines.append("## 7. Limitations")
lines.append("")
lines.append(
    "1. The target is an OSM-based urban-activity proxy, not measured "
    "EV charging demand."
)
lines.append(
    "2. Permutation importance describes predictive dependence, "
    "not causal influence."
)
lines.append(
    "3. Correlated predictors can share or mask importance."
)
lines.append(
    "4. Spatial validation within Colombo does not establish "
    "generalization to other cities."
)
lines.append(
    "5. External validation using actual EV charging-session data "
    "is required before operational demand-planning claims."
)
lines.append("")
lines.append("## 8. Conclusion")
lines.append("")
lines.append(
    "The explainability experiments show that geographic position "
    "is a major predictor of OSM-based demand proxy presence, while "
    "road-network characteristics provide additional predictive "
    "information. These findings support further investigation using "
    "real EV charging data and spatial graph-learning methods."
)

output_path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("EXPLAINABILITY REPORT CREATED")
print("Output:", output_path)
print("Report lines:", len(lines))
print("Geographic ROC-AUC difference:", round(difference, 4))
