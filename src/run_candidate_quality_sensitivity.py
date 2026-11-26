from pathlib import Path
import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]

DEMAND_PATH = ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
CANDIDATE_PATH = ROOT / "data/processed/candidates/candidate_locations.csv"
DISTANCE_PATH = ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"

OUTPUT_DIR = ROOT / "outputs/optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_CANDIDATES = {"C0195", "C0198"}
P_VALUES = [5, 10]
SEPARATIONS = [0.0, 0.25, 0.5, 1.0, 1.5]
RADIUS_KM = 3.0
EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(candidates):
    lat = np.radians(candidates["latitude"].to_numpy(dtype=float))
    lon = np.radians(candidates["longitude"].to_numpy(dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :])
        * np.sin(dlon / 2) ** 2
    )

    distances = 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )
    np.fill_diagonal(distances, np.inf)
    return distances


def main():
    demand = pd.read_csv(DEMAND_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH, dtype={"candidate_id": str})
    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)

    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    excluded_present = EXCLUDED_CANDIDATES.intersection(
        candidates["candidate_id"]
    )

    print("Original candidates:", len(candidates))
    print("Excluded candidates:", sorted(excluded_present))

    candidates = candidates[
        ~candidates["candidate_id"].isin(EXCLUDED_CANDIDATES)
    ].copy()

    if candidates.empty:
        raise ValueError("No eligible candidates remain.")

    candidate_ids = candidates["candidate_id"].tolist()
    missing = set(candidate_ids) - set(distance_df.columns)

    if missing:
        raise ValueError(f"Missing distance columns: {sorted(missing)}")

    D = distance_df[candidate_ids].to_numpy(dtype=float)
    weights = demand["demand_weight"].to_numpy(dtype=float)

    if D.shape[0] != len(weights):
        raise ValueError("Demand and distance matrix rows do not align.")

    if not np.isfinite(D).any():
        raise ValueError("No reachable demand-candidate pairs.")

    candidate_distances = haversine_matrix(candidates)

    print("Eligible candidates:", len(candidates))
    print("Demand nodes:", len(demand))
    print("Demand proxy weight:", weights.sum())
    print("Reachable pairs:", int(np.isfinite(D).sum()))

    results = []
    selections = []

    for p in P_VALUES:
        for separation in SEPARATIONS:
            print(f"\nP={p}, separation={separation} km")

            model = pulp.LpProblem(
                f"Filtered_MCLP_P{p}_S{separation}",
                pulp.LpMaximize
            )

            y = {
                j: pulp.LpVariable(f"y_{j}", cat="Binary")
                for j in range(len(candidate_ids))
            }
            z = {
                i: pulp.LpVariable(f"z_{i}", cat="Binary")
                for i in range(len(weights))
            }

            model += pulp.lpSum(
                weights[i] * z[i] for i in range(len(weights))
            )

            model += pulp.lpSum(y.values()) == p

            for i in range(len(weights)):
                reachable = np.flatnonzero(
                    np.isfinite(D[i]) & (D[i] <= RADIUS_KM)
                )
                model += z[i] <= pulp.lpSum(y[j] for j in reachable)

            if separation > 0:
                rows, cols = np.where(
                    np.triu(candidate_distances < separation, k=1)
                )
                for j, k in zip(rows, cols):
                    model += y[j] + y[k] <= 1

            solver = pulp.PULP_CBC_CMD(
                msg=False,
                timeLimit=300,
                gapRel=0.01
            )

            model.solve(solver)
            status = pulp.LpStatus[model.status]

            result = {
                "scenario": "candidate_quality_filtered",
                "p": p,
                "coverage_radius_km": RADIUS_KM,
                "minimum_separation_km": separation,
                "solver_status": status,
                "covered_proxy_weight": np.nan,
                "coverage_percentage": np.nan,
                "selected_facilities": 0
            }

            if status == "Optimal":
                objective = pulp.value(model.objective)
                selected_indices = [
                    j for j in range(len(candidate_ids))
                    if pulp.value(y[j]) is not None
                    and pulp.value(y[j]) > 0.5
                ]

                result["covered_proxy_weight"] = float(objective)
                result["coverage_percentage"] = (
                    float(objective) / weights.sum() * 100
                )
                result["selected_facilities"] = len(selected_indices)

                for j in selected_indices:
                    row = candidates.iloc[j]
                    selections.append({
                        "p": p,
                        "coverage_radius_km": RADIUS_KM,
                        "minimum_separation_km": separation,
                        "candidate_id": candidate_ids[j],
                        "node_id": row["node_id"],
                        "longitude": row["longitude"],
                        "latitude": row["latitude"]
                    })

                print(
                    f"Optimal | Coverage: {objective:.0f}/"
                    f"{weights.sum():.0f} "
                    f"({result['coverage_percentage']:.3f}%)"
                )
                print("Selected:", [
                    candidate_ids[j] for j in selected_indices
                ])
            else:
                print("Solver status:", status)

            results.append(result)

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "mclp_candidate_quality_sensitivity.csv",
        index=False
    )

    pd.DataFrame(selections).to_csv(
        OUTPUT_DIR / "mclp_candidate_quality_selected_locations.csv",
        index=False
    )

    print("\nCandidate-quality sensitivity completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
