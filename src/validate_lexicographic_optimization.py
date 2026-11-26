from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEMAND = pd.read_csv(
    ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
)
CANDIDATES = pd.read_csv(
    ROOT / "data/processed/candidates/candidate_locations.csv",
    dtype={"candidate_id": str}
)
DISTANCES = pd.read_csv(
    ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv",
    index_col=0
)
RESULTS = pd.read_csv(
    ROOT / "outputs/optimization/mclp_lexicographic_optimization_results.csv"
)
SELECTED = pd.read_csv(
    ROOT / "outputs/optimization/mclp_lexicographic_selected_facilities.csv",
    dtype={"candidate_id": str}
)

DISTANCES.index = DISTANCES.index.astype(str)
DISTANCES.columns = DISTANCES.columns.astype(str)

weights = DEMAND["demand_weight"].to_numpy(dtype=float)
total_weight = weights.sum()

EARTH_RADIUS_KM = 6371.0088
validation = []

for _, result in RESULTS.iterrows():
    scenario = result["scenario"]
    p = int(result["p"])

    chosen = SELECTED[
        (SELECTED["scenario"] == scenario) &
        (SELECTED["p"] == p)
    ].copy()

    ids = chosen["candidate_id"].tolist()

    if len(ids) != p or len(set(ids)) != p:
        raise ValueError(f"Incorrect facility count or duplicate IDs: {scenario}, P={p}")

    if not set(ids).issubset(set(CANDIDATES["candidate_id"])):
        raise ValueError("Selected candidate missing from original candidate data.")

    # Recalculate actual coverage from the original road-distance matrix.
    D = DISTANCES[ids].to_numpy(dtype=float)
    covered = np.any(np.isfinite(D) & (D <= 3.0), axis=1)
    actual_covered_weight = float(weights[covered].sum())
    actual_coverage_pct = actual_covered_weight / total_weight * 100

    # Check straight-line minimum facility separation.
    lat = np.radians(chosen["latitude"].to_numpy(dtype=float))
    lon = np.radians(chosen["longitude"].to_numpy(dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :])
        * np.sin(dlon / 2) ** 2
    )

    separation = 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )
    np.fill_diagonal(separation, np.inf)
    min_separation = float(separation.min())

    # Recheck candidate eligibility.
    original = CANDIDATES.set_index("candidate_id").loc[ids]
    excluded_network = {"C0195", "C0198"}
    network_ok = (
        scenario != "combined_quality"
        or not set(ids).intersection(excluded_network)
    )
    snap_ok = (
        scenario != "combined_quality"
        or bool((original["max_snap_distance_m"] <= 100).all())
    )

    actual_snap_total = float(original["mean_snap_distance_m"].sum())
    reported_coverage = float(result["covered_proxy_weight"])

    coverage_ok = actual_covered_weight >= reported_coverage - 1e-6
    separation_ok = min_separation >= 0.5 - 1e-8
    count_ok = len(ids) == p
    quality_ok = network_ok and snap_ok

    passed = coverage_ok and separation_ok and count_ok and quality_ok

    validation.append({
        "scenario": scenario,
        "p": p,
        "selected_count": len(ids),
        "reported_stage1_coverage": reported_coverage,
        "recalculated_stage2_coverage": actual_covered_weight,
        "actual_coverage_percentage": actual_coverage_pct,
        "minimum_separation_km": min_separation,
        "actual_total_mean_snap_distance_m": actual_snap_total,
        "coverage_check": coverage_ok,
        "separation_check": separation_ok,
        "quality_check": quality_ok,
        "validation": "PASS" if passed else "FAIL"
    })

    print(
        f"{scenario}, P={p}: "
        f"coverage={actual_covered_weight:.0f}/{total_weight:.0f} "
        f"({actual_coverage_pct:.3f}%), "
        f"minimum separation={min_separation:.3f} km, "
        f"validation={'PASS' if passed else 'FAIL'}"
    )

output = pd.DataFrame(validation)
path = ROOT / "outputs/analysis/lexicographic_optimization_validation.csv"
output.to_csv(path, index=False)

print("\n========== VALIDATION SUMMARY ==========")
print(output.to_string(index=False))
print("\nPassed:", (output["validation"] == "PASS").sum(), "/", len(output))
print("Saved:", path)

if not (output["validation"] == "PASS").all():
    raise SystemExit("Validation failed. Review the results before proceeding.")
