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

P_VALUES = [5, 10]
RADIUS_KM = 3.0
MIN_SEPARATION_KM = 0.5
TIME_LIMIT = 300
RELATIVE_GAP = 0.01

NETWORK_EXCLUSIONS = {"C0195", "C0198"}


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

    distances = 2 * 6371.0088 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    np.fill_diagonal(distances, np.inf)
    return distances


def build_model(demand, candidates, D, p, stage, coverage_target=None):
    weights = demand["demand_weight"].to_numpy(dtype=float)
    snap_costs = candidates["mean_snap_distance_m"].to_numpy(dtype=float)

    n_demand = len(weights)
    n_candidates = len(candidates)

    model = pulp.LpProblem(
        f"Lexicographic_MCLP_P{p}_{stage}",
        pulp.LpMaximize if stage == "coverage" else pulp.LpMinimize
    )

    y = {
        j: pulp.LpVariable(f"y_{j}", cat="Binary")
        for j in range(n_candidates)
    }

    z = {
        i: pulp.LpVariable(f"z_{i}", cat="Binary")
        for i in range(n_demand)
    }

    coverage = pulp.lpSum(
        weights[i] * z[i] for i in range(n_demand)
    )

    if stage == "coverage":
        model += coverage
    else:
        model += pulp.lpSum(
            snap_costs[j] * y[j] for j in range(n_candidates)
        )
        model += coverage >= coverage_target

    model += pulp.lpSum(y.values()) == p

    for i in range(n_demand):
        reachable = np.flatnonzero(
            np.isfinite(D[i]) & (D[i] <= RADIUS_KM)
        )

        if len(reachable) == 0:
            model += z[i] == 0
        else:
            model += z[i] <= pulp.lpSum(y[j] for j in reachable)

    # Enforce minimum straight-line separation between selected facilities.
    distances = haversine_matrix(candidates)
    rows, cols = np.where(
        np.triu(distances < MIN_SEPARATION_KM, k=1)
    )

    for j, k in zip(rows, cols):
        model += y[j] + y[k] <= 1

    return model, y, z


def solve(model):
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=TIME_LIMIT,
        gapRel=RELATIVE_GAP
    )
    model.solve(solver)
    return pulp.LpStatus[model.status]


def main():
    print("Loading datasets...")

    demand = pd.read_csv(DEMAND_PATH)
    candidates_all = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"candidate_id": str}
    )

    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)
    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    candidates_all["candidate_id"] = (
        candidates_all["candidate_id"].astype(str)
    )

    if candidates_all["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found.")

    scenarios = {
        "baseline": set(),
        "combined_quality": NETWORK_EXCLUSIONS | set(
            candidates_all.loc[
                candidates_all["max_snap_distance_m"] > 100,
                "candidate_id"
            ]
        )
    }

    results = []
    selections = []

    total_weight = float(demand["demand_weight"].sum())

    print("Demand nodes:", len(demand))
    print("Total OSM-based demand-proxy weight:", total_weight)

    for scenario, excluded in scenarios.items():
        candidates = candidates_all[
            ~candidates_all["candidate_id"].isin(excluded)
        ].copy().reset_index(drop=True)

        candidate_ids = candidates["candidate_id"].tolist()

        missing = set(candidate_ids) - set(distance_df.columns)
        if missing:
            raise ValueError(
                f"Missing distance columns: {sorted(missing)}"
            )

        D = distance_df[candidate_ids].to_numpy(dtype=float)

        if D.shape[0] != len(demand):
            raise ValueError("Demand and distance rows do not align.")

        print(
            f"\n========== SCENARIO: {scenario} =========="
        )
        print("Eligible candidates:", len(candidates))

        for p in P_VALUES:
            print(f"\nP={p}: Stage 1 - maximize coverage")

            model1, y1, z1 = build_model(
                demand, candidates, D, p, "coverage"
            )

            status1 = solve(model1)

            if status1 != "Optimal":
                print("Stage 1 status:", status1)
                results.append({
                    "scenario": scenario,
                    "p": p,
                    "stage1_status": status1,
                    "stage2_status": "Not run",
                    "covered_proxy_weight": np.nan,
                    "coverage_percentage": np.nan,
                    "total_selected_mean_snap_distance_m": np.nan,
                })
                continue

            best_coverage = float(pulp.value(model1.objective))

            print(
                f"Stage 1: {status1}, coverage "
                f"{best_coverage:.0f}/{total_weight:.0f} "
                f"({best_coverage / total_weight * 100:.3f}%)"
            )

            print("Stage 2 - minimize selected facility snap distance")

            model2, y2, z2 = build_model(
                demand,
                candidates,
                D,
                p,
                "quality",
                coverage_target=best_coverage
            )

            status2 = solve(model2)

            result = {
                "scenario": scenario,
                "p": p,
                "eligible_candidates": len(candidates),
                "coverage_radius_km": RADIUS_KM,
                "minimum_separation_km": MIN_SEPARATION_KM,
                "stage1_status": status1,
                "stage2_status": status2,
                "covered_proxy_weight": best_coverage,
                "total_proxy_weight": total_weight,
                "coverage_percentage": (
                    best_coverage / total_weight * 100
                ),
                "total_selected_mean_snap_distance_m": np.nan,
                "selected_facilities": 0,
            }

            if status2 == "Optimal":
                selected_indices = [
                    j for j in range(len(candidate_ids))
                    if y2[j].value() is not None
                    and y2[j].value() > 0.5
                ]

                selected_snap_total = sum(
                    candidates.iloc[j]["mean_snap_distance_m"]
                    for j in selected_indices
                )

                result["total_selected_mean_snap_distance_m"] = (
                    float(selected_snap_total)
                )
                result["selected_facilities"] = len(selected_indices)

                for j in selected_indices:
                    row = candidates.iloc[j]

                    selections.append({
                        "scenario": scenario,
                        "p": p,
                        "candidate_id": candidate_ids[j],
                        "node_id": row["node_id"],
                        "longitude": row["longitude"],
                        "latitude": row["latitude"],
                        "mean_snap_distance_m": row["mean_snap_distance_m"],
                        "max_snap_distance_m": row["max_snap_distance_m"],
                    })

                print(
                    f"Stage 2: {status2}, total selected mean "
                    f"snap distance={selected_snap_total:.2f} m"
                )
                print("Selected:", [
                    candidate_ids[j] for j in selected_indices
                ])
            else:
                print("Stage 2 status:", status2)

            results.append(result)

    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)

    results_path = (
        OUTPUT_DIR / "mclp_lexicographic_optimization_results.csv"
    )
    selections_path = (
        OUTPUT_DIR / "mclp_lexicographic_selected_facilities.csv"
    )

    results_df.to_csv(results_path, index=False)
    selections_df.to_csv(selections_path, index=False)

    print("\n========== FINAL RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nSaved:", results_path)
    print("Saved:", selections_path)


if __name__ == "__main__":
    main()
