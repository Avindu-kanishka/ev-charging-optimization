from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pulp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMAND_PATH = PROJECT_ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
CANDIDATE_PATH = PROJECT_ROOT / "data/processed/candidates/candidate_locations.csv"
DISTANCE_PATH = PROJECT_ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/optimization"

# Exploratory assumptions, not statutory or planning standards.
SEPARATION_THRESHOLDS_KM = [0.0, 0.25, 0.5, 1.0, 1.5]
P_VALUES = [5, 10]
COVERAGE_RADIUS_KM = 3.0
TIME_LIMIT_SECONDS = 180
RELATIVE_GAP = 0.01
EARTH_RADIUS_KM = 6371.0088


def haversine_matrix_km(candidates):
    """Return straight-line candidate-to-candidate distances in kilometres."""
    lat = np.radians(candidates["latitude"].to_numpy(dtype=float))
    lon = np.radians(candidates["longitude"].to_numpy(dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :])
        * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    np.fill_diagonal(distances, np.inf)
    return distances


def load_inputs():
    demand = pd.read_csv(DEMAND_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)
    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)

    required_demand = {"node_id", "demand_weight"}
    required_candidates = {"candidate_id", "latitude", "longitude"}
    if not required_demand.issubset(demand.columns):
        raise ValueError(f"Demand file must contain {sorted(required_demand)}")
    if not required_candidates.issubset(candidates.columns):
        raise ValueError(
            f"Candidate file must contain {sorted(required_candidates)}"
        )

    demand["node_id"] = demand["node_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    if demand["node_id"].duplicated().any():
        raise ValueError("Duplicate demand node IDs found.")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found.")

    demand = demand.set_index("node_id")
    candidates = candidates.set_index("candidate_id")

    missing_demand = distance_df.index.difference(demand.index)
    missing_candidates = distance_df.columns.difference(candidates.index)
    if len(missing_demand) or len(missing_candidates):
        raise ValueError(
            "Distance matrix labels do not align with demand/candidate files: "
            f"{len(missing_demand)} missing demand, "
            f"{len(missing_candidates)} missing candidates."
        )

    demand = demand.loc[distance_df.index]
    candidates = candidates.loc[distance_df.columns]

    D = distance_df.to_numpy(dtype=float)
    weights = demand["demand_weight"].to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Demand weights must be finite and non-negative.")
    if not np.isfinite(D).any():
        raise ValueError("Distance matrix contains no reachable pairs.")

    return demand, candidates, D, weights, distance_df.index.tolist(), distance_df.columns.tolist()


def solve_scenario(D, weights, demand_ids, candidate_ids, candidates,
                   candidate_separation, p, radius_km):
    n_demand, n_candidates = D.shape
    model = pulp.LpProblem(
        f"MCLP_P{p}_R{radius_km}_S{candidate_separation}",
        pulp.LpMaximize,
    )

    y = {
        j: pulp.LpVariable(f"facility_{j}", cat=pulp.LpBinary)
        for j in range(n_candidates)
    }
    z = {
        i: pulp.LpVariable(f"covered_{i}", cat=pulp.LpBinary)
        for i in range(n_demand)
    }

    model += pulp.lpSum(weights[i] * z[i] for i in range(n_demand))
    model += pulp.lpSum(y[j] for j in range(n_candidates)) == p

    for i in range(n_demand):
        reachable = np.flatnonzero(
            np.isfinite(D[i]) & (D[i] <= radius_km)
        )
        model += z[i] <= pulp.lpSum(y[j] for j in reachable)

    separation_pairs = []
    if candidate_separation > 0:
        candidate_distances = haversine_matrix_km(candidates)
        rows, cols = np.where(
            np.triu(candidate_distances < candidate_separation, k=1)
        )
        separation_pairs = list(zip(rows.tolist(), cols.tolist()))
        for pair_number, (j, k) in enumerate(separation_pairs):
            model += y[j] + y[k] <= 1, f"separation_{pair_number}"

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=TIME_LIMIT_SECONDS,
        gapRel=RELATIVE_GAP,
    )
    model.solve(solver)
    status = pulp.LpStatus[model.status]

    result = {
        "method": "MCLP_minimum_separation_sensitivity",
        "p": p,
        "coverage_radius_km": radius_km,
        "minimum_separation_km": candidate_separation,
        "solver_status": status,
        "separation_pair_constraints": len(separation_pairs),
        "covered_proxy_weight": np.nan,
        "total_proxy_weight": float(weights.sum()),
        "coverage_percentage": np.nan,
        "selected_facilities": 0,
    }
    selected_rows = []

    # Report comparable coverage only for a proven optimal solution.
    if status == "Optimal":
        objective = pulp.value(model.objective)
        selected_indices = [
            j for j in range(n_candidates)
            if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5
        ]
        if objective is not None:
            result["covered_proxy_weight"] = float(objective)
            result["coverage_percentage"] = (
                float(objective) / weights.sum() * 100.0
                if weights.sum() > 0 else np.nan
            )
        result["selected_facilities"] = len(selected_indices)

        for j in selected_indices:
            candidate_id = candidate_ids[j]
            row = candidates.iloc[j]
            selected_rows.append({
                "p": p,
                "coverage_radius_km": radius_km,
                "minimum_separation_km": candidate_separation,
                "candidate_id": candidate_id,
                "node_id": row.get("node_id", ""),
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "candidate_sources": row.get("candidate_sources", ""),
            })

    return result, selected_rows


def main():
    print("=" * 72)
    print("EVISION AI - MCLP FACILITY SEPARATION SENSITIVITY")
    print("=" * 72)
    print("Demand source: non-candidate OSM POI sensitivity scenario")
    print("Coverage distance: road-network distance")
    print("Facility separation: straight-line (Haversine) distance")
    print(f"P values: {P_VALUES}")
    print(f"Coverage radius: {COVERAGE_RADIUS_KM} km")
    print(f"Separation thresholds: {SEPARATION_THRESHOLDS_KM} km")
    print("Thresholds are exploratory assumptions, not planning standards.")

    _, candidates, D, weights, demand_ids, candidate_ids = load_inputs()
    print(f"Demand nodes: {len(demand_ids):,}")
    print(f"Candidate locations: {len(candidate_ids):,}")
    print(f"Total demand proxy weight: {weights.sum():,.0f}")

    results = []
    selected_locations = []
    total_runs = len(P_VALUES) * len(SEPARATION_THRESHOLDS_KM)
    run_number = 0

    for p in P_VALUES:
        for separation in SEPARATION_THRESHOLDS_KM:
            run_number += 1
            print(
                f"\n[{run_number}/{total_runs}] "
                f"P={p}, radius={COVERAGE_RADIUS_KM} km, "
                f"minimum separation={separation} km"
            )
            result, selected = solve_scenario(
                D, weights, demand_ids, candidate_ids, candidates,
                separation, p, COVERAGE_RADIUS_KM,
            )
            results.append(result)
            selected_locations.extend(selected)
            print(
                f"Status: {result['solver_status']}; "
                f"separation constraints: "
                f"{result['separation_pair_constraints']}"
            )
            if result["solver_status"] == "Optimal":
                print(
                    f"Coverage: {result['covered_proxy_weight']:.0f}/"
                    f"{result['total_proxy_weight']:.0f} "
                    f"({result['coverage_percentage']:.3f}%)"
                )
                print(
                    f"Selected facilities: {result['selected_facilities']}"
                )
            else:
                print("No optimal result; coverage is not reported.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "mclp_facility_separation_sensitivity.csv"
    selected_path = OUTPUT_DIR / "mclp_facility_separation_selected_locations.csv"

    pd.DataFrame(results).to_csv(results_path, index=False)
    pd.DataFrame(selected_locations).to_csv(selected_path, index=False)

    print("\nSensitivity experiment completed.")
    print(f"Results: {results_path}")
    print(f"Selected locations: {selected_path}")
    print(
        "\nInterpretation: facility separation is a scenario constraint. "
        "It does not establish a statutory or universally optimal spacing rule."
    )


if __name__ == "__main__":
    main()
