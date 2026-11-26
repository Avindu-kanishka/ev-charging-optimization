import pandas as pd

selection_path = "outputs/optimization/mclp_multiobjective_candidate_selections.csv"
metrics_path = "outputs/analysis/candidate_road_accessibility_metrics.csv"

selected = pd.read_csv(selection_path)
metrics = pd.read_csv(metrics_path)

quality_columns = [
    "mean_snap_distance_m",
    "max_snap_distance_m",
    "in_degree",
    "out_degree",
    "weak_component_size",
    "strong_component_size",
]

# Remove duplicate quality columns from the selection file.
selected = selected.drop(
    columns=[c for c in quality_columns if c in selected.columns]
)

df = selected.merge(
    metrics[["candidate_id"] + quality_columns],
    on="candidate_id",
    how="left",
    validate="many_to_one"
)

if df[quality_columns].isna().any().any():
    raise ValueError("Missing candidate quality metrics after merging.")

print("========== SELECTED FACILITY QUALITY ==========")

summary = df.groupby(["scenario", "p"]).agg(
    selected_count=("candidate_id", "count"),
    mean_snap_distance=("mean_snap_distance_m", "mean"),
    maximum_snap_distance=("max_snap_distance_m", "max"),
    mean_in_degree=("in_degree", "mean"),
    mean_out_degree=("out_degree", "mean"),
    minimum_weak_component=("weak_component_size", "min"),
    minimum_strong_component=("strong_component_size", "min"),
).reset_index()

print(summary.to_string(index=False))

df["isolated_component"] = df["weak_component_size"] < 100
df["high_snap_distance"] = df["max_snap_distance_m"] > 100
df["zero_in_degree"] = df["in_degree"] == 0
df["zero_out_degree"] = df["out_degree"] == 0

print("\n========== SELECTED SITES WITH QUALITY FLAGS ==========")

print(
    df[
        [
            "scenario", "p", "candidate_id",
            "mean_snap_distance_m", "max_snap_distance_m",
            "in_degree", "out_degree",
            "weak_component_size", "strong_component_size",
            "isolated_component", "high_snap_distance",
            "zero_in_degree", "zero_out_degree"
        ]
    ].to_string(index=False)
)

print("\n========== FLAG COUNTS ==========")

print(
    df.groupby(["scenario", "p"])[
        [
            "isolated_component",
            "high_snap_distance",
            "zero_in_degree",
            "zero_out_degree"
        ]
    ].sum().to_string()
)

output_path = "outputs/analysis/selected_facility_quality_summary.csv"
summary.to_csv(output_path, index=False)

print("\nSaved:", output_path)
