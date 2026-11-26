from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEMAND_PATH = ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
DISTANCE_PATH = ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"
SELECTION_PATH = ROOT / "outputs/optimization/mclp_lexicographic_selected_facilities.csv"

OUTPUT_DIR = ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADII_KM = [1.0, 2.0, 3.0]


def main():
    print("Loading demand and road-network distances...")

    demand = pd.read_csv(DEMAND_PATH)
    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)
    selections = pd.read_csv(
        SELECTION_PATH,
        dtype={"candidate_id": str}
    )

    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    if len(demand) != len(distance_df):
        raise ValueError("Demand and distance matrix row counts differ.")

    if demand["node_id"].astype(str).tolist() != distance_df.index.tolist():
        raise ValueError("Demand node order does not match distance matrix.")

    if demand["node_id"].duplicated().any():
        raise ValueError("Duplicate demand node IDs found.")

    weights = demand["demand_weight"].to_numpy(dtype=float)
    total_weight = float(weights.sum())
    total_nodes = len(demand)

    summaries = []
    uncovered_records = []

    print("Demand nodes:", total_nodes)
    print("Total OSM-based demand-proxy weight:", total_weight)

    configurations = [
        ("baseline", 5),
        ("baseline", 10),
        ("combined_quality", 5),
        ("combined_quality", 10),
    ]

    for scenario, p in configurations:
        chosen = selections[
            (selections["scenario"] == scenario) &
            (selections["p"] == p)
        ]

        candidate_ids = chosen["candidate_id"].astype(str).tolist()

        if len(candidate_ids) != p:
            raise ValueError(
                f"{scenario}, P={p}: expected {p} facilities, "
                f"found {len(candidate_ids)}"
            )

        missing = set(candidate_ids) - set(distance_df.columns)
        if missing:
            raise ValueError(
                f"Missing distance columns: {sorted(missing)}"
            )

        selected_distances = distance_df[candidate_ids].to_numpy(
            dtype=float
        )

        nearest_distance = np.min(selected_distances, axis=1)

        print(f"\n{scenario.upper()} | P={p}")
        print("Selected facilities:", candidate_ids)

        for radius in RADII_KM:
            covered = (
                np.isfinite(nearest_distance)
                & (nearest_distance <= radius)
            )

            covered_nodes = int(covered.sum())
            covered_weight = float(weights[covered].sum())

            uncovered = ~covered
            uncovered_nodes = int(uncovered.sum())
            uncovered_weight = float(weights[uncovered].sum())

            node_pct = covered_nodes / total_nodes * 100
            weight_pct = covered_weight / total_weight * 100

            summaries.append({
                "scenario": scenario,
                "p": p,
                "coverage_radius_km": radius,
                "selected_facilities": len(candidate_ids),
                "total_demand_nodes": total_nodes,
                "covered_demand_nodes": covered_nodes,
                "uncovered_demand_nodes": uncovered_nodes,
                "demand_node_coverage_pct": node_pct,
                "total_proxy_weight": total_weight,
                "covered_proxy_weight": covered_weight,
                "uncovered_proxy_weight": uncovered_weight,
                "proxy_weight_coverage_pct": weight_pct,
            })

            print(
                f"  Radius {radius:.0f} km: "
                f"{covered_nodes}/{total_nodes} nodes "
                f"({node_pct:.2f}%), "
                f"proxy weight {covered_weight:.0f}/{total_weight:.0f} "
                f"({weight_pct:.3f}%)"
            )

            for i in np.flatnonzero(uncovered):
                row = demand.iloc[i]

                uncovered_records.append({
                    "scenario": scenario,
                    "p": p,
                    "coverage_radius_km": radius,
                    "node_id": row["node_id"],
                    "longitude": row["longitude"],
                    "latitude": row["latitude"],
                    "demand_weight": row["demand_weight"],
                    "proxy_count": row["proxy_count"],
                    "nearest_selected_facility_distance_km": (
                        nearest_distance[i]
                        if np.isfinite(nearest_distance[i])
                        else np.nan
                    )
                })

    summary_df = pd.DataFrame(summaries)
    uncovered_df = pd.DataFrame(uncovered_records)

    summary_path = OUTPUT_DIR / "spatial_accessibility_coverage_summary.csv"
    uncovered_path = OUTPUT_DIR / "underserved_demand_locations.csv"

    summary_df.to_csv(summary_path, index=False)
    uncovered_df.to_csv(uncovered_path, index=False)

    print("\n========== ACCESSIBILITY SUMMARY ==========")
    print(
        summary_df[
            [
                "scenario",
                "p",
                "coverage_radius_km",
                "covered_demand_nodes",
                "uncovered_demand_nodes",
                "demand_node_coverage_pct",
                "covered_proxy_weight",
                "proxy_weight_coverage_pct",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", summary_path)
    print("Saved:", uncovered_path)
    print("\nSpatial accessibility analysis completed.")


if __name__ == "__main__":
    main()
