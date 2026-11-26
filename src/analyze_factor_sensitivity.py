from pathlib import Path
from itertools import combinations
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "outputs/optimization/"
    "mclp_demand_weight_radius_selected_locations.csv"
)

OUTPUT_DIR = ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, dtype={"candidate_id": str})

rows = []

def compare_sets(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


# 1. Compare weighting methods at fixed P and radius
for (p, radius), group in df.groupby(
    ["p", "coverage_radius_km"]
):
    scenarios = {
        weight: set(part["candidate_id"])
        for weight, part in group.groupby("weight_scenario")
    }

    for (name_a, set_a), (name_b, set_b) in combinations(
        scenarios.items(), 2
    ):
        rows.append({
            "comparison_type": "weighting_sensitivity",
            "p": p,
            "fixed_radius_km": radius,
            "fixed_weight_scenario": "all",
            "scenario_a": name_a,
            "scenario_b": name_b,
            "jaccard_similarity": compare_sets(set_a, set_b)
        })


# 2. Compare coverage radii at fixed P and weighting method
for (p, weight), group in df.groupby(
    ["p", "weight_scenario"]
):
    scenarios = {
        radius: set(part["candidate_id"])
        for radius, part in group.groupby("coverage_radius_km")
    }

    for (radius_a, set_a), (radius_b, set_b) in combinations(
        scenarios.items(), 2
    ):
        rows.append({
            "comparison_type": "radius_sensitivity",
            "p": p,
            "fixed_radius_km": "all",
            "fixed_weight_scenario": weight,
            "scenario_a": radius_a,
            "scenario_b": radius_b,
            "jaccard_similarity": compare_sets(set_a, set_b)
        })


results = pd.DataFrame(rows)

output = OUTPUT_DIR / "mclp_factor_sensitivity_comparison.csv"
results.to_csv(output, index=False)

print("COMPARISON COUNTS")
print(results.groupby(
    ["comparison_type", "p"]
).size().to_string())

print("\nAVERAGE JACCARD SIMILARITY")
print(results.groupby(
    ["comparison_type", "p"]
)["jaccard_similarity"].agg(
    ["count", "mean", "median", "min", "max"]
).to_string())

print("\nRESULTS SAVED:", output)
print("\nFactor sensitivity comparison completed.")
