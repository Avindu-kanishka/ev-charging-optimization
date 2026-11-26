from pathlib import Path
from itertools import combinations
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT = (
    ROOT / "outputs/optimization/"
    "mclp_demand_weight_radius_selected_locations.csv"
)

selected = pd.read_csv(INPUT, dtype={"candidate_id": str})

required = {
    "weight_scenario",
    "p",
    "coverage_radius_km",
    "candidate_id"
}

missing = required - set(selected.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

scenario_cols = [
    "weight_scenario",
    "p",
    "coverage_radius_km"
]

scenario_sets = (
    selected.groupby(scenario_cols)["candidate_id"]
    .apply(set)
)

if len(scenario_sets) != 32:
    raise ValueError(
        f"Expected 32 scenarios, found {len(scenario_sets)}"
    )

frequency_rows = []

for p in sorted(selected["p"].unique()):
    p_data = selected[selected["p"] == p]

    scenarios = (
        p_data.groupby(
            ["weight_scenario", "coverage_radius_km"]
        )["candidate_id"]
        .apply(set)
    )

    total_scenarios = len(scenarios)

    counts = {}

    for facilities in scenarios:
        for candidate in facilities:
            counts[candidate] = counts.get(candidate, 0) + 1

    for candidate, count in counts.items():
        frequency_rows.append({
            "p": int(p),
            "candidate_id": candidate,
            "selected_scenarios": count,
            "total_scenarios": total_scenarios,
            "selection_frequency_percentage":
                count / total_scenarios * 100
        })

frequency_df = pd.DataFrame(frequency_rows)

frequency_df = frequency_df.sort_values(
    ["p", "selected_scenarios", "candidate_id"],
    ascending=[True, False, True]
)

frequency_path = (
    OUTPUT_DIR / "mclp_candidate_selection_stability.csv"
)

frequency_df.to_csv(frequency_path, index=False)

jaccard_rows = []

for p in sorted(selected["p"].unique()):
    p_data = selected[selected["p"] == p]

    scenarios = {
        (weight, radius): facilities
        for (weight, radius), facilities in
        p_data.groupby(
            ["weight_scenario", "coverage_radius_km"]
        )["candidate_id"].apply(set).items()
    }

    for (scenario_a, facilities_a), (
        scenario_b, facilities_b
    ) in combinations(scenarios.items(), 2):

        intersection = facilities_a & facilities_b
        union = facilities_a | facilities_b

        similarity = (
            len(intersection) / len(union)
            if union else 1.0
        )

        jaccard_rows.append({
            "p": int(p),
            "scenario_a_weight": scenario_a[0],
            "scenario_a_radius_km": scenario_a[1],
            "scenario_b_weight": scenario_b[0],
            "scenario_b_radius_km": scenario_b[1],
            "common_facilities": len(intersection),
            "union_facilities": len(union),
            "jaccard_similarity": similarity
        })

jaccard_df = pd.DataFrame(jaccard_rows)

jaccard_path = (
    OUTPUT_DIR / "mclp_scenario_jaccard_similarity.csv"
)

jaccard_df.to_csv(jaccard_path, index=False)

print("========== CANDIDATE SELECTION STABILITY ==========")

for p in sorted(frequency_df["p"].unique()):
    subset = frequency_df[frequency_df["p"] == p]

    print(f"\nP={p}")
    print("Unique selected candidates:", len(subset))
    print("Total scenarios:", int(
        subset["total_scenarios"].iloc[0]
    ))

    print("\nMost frequently selected candidates:")
    print(
        subset.head(10)[
            [
                "candidate_id",
                "selected_scenarios",
                "selection_frequency_percentage"
            ]
        ].to_string(index=False)
    )

print("\n========== JACCARD SIMILARITY ==========")

print(
    jaccard_df.groupby("p")["jaccard_similarity"]
    .agg(["count", "mean", "median", "min", "max"])
    .to_string()
)

print("\nSaved:")
print(frequency_path)
print(jaccard_path)

print("\nCandidate stability analysis completed.")
