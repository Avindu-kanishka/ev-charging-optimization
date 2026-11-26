from pathlib import Path

import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]

DEMAND_PATH = ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
CANDIDATE_PATH = ROOT / "data/processed/candidates/candidate_locations.csv"
DISTANCE_PATH = ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"
METRICS_PATH = ROOT / "outputs/analysis/candidate_road_accessibility_metrics.csv"

OUTPUT_DIR = ROOT / "outputs/optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES = [5, 10]
RADIUS_KM = 3.0
MIN_SEPARATION_KM = 0.5
TIME_LIMIT = 300
RELATIVE_GAP = 0.01

NETWORK_EXCLUSIONS = {"C0195"}
STRICT_NETWORK_EXCLUSIONS = {"C0195", "C0198"}


def build_scenarios(candidates):
    snap_exclusions = set(
        candidates.loc[
            candidates["max_snap_distance_m"] > 100,
            "candidate_id"
        ]
    )

    scenarios = {
        "baseline": set(),
        "network_connectivity": NETWORK_EXCLUSIONS,
        "strict_network_quality": STRICT_NETWORK_EXCLUSIONS,
        "snapping_quality": snap_exclusions,
        "combined_quality": snap_exclusions | NETWORK_EXCLUSIONS,
    }

    expected_counts = {
        "baseline": 294,
        "network_connectivity": 293,
        "strict_network_quality": 292,
        "snapping_quality": 257,
        "combined_quality": 256,
    }

    print("\nCandidate scenario definitions:")
    for name, excluded in scenarios.items():
        eligible_count = len(candidates) - len(excluded)
        print(
            f"{name}: {eligible_count} eligible, "
            f"{len(excluded)} excluded"
        )

        if eligible_count != expected_counts[name]:
            raise ValueError(
                f"{name}: expected {expected_counts[name]} candidates, "
                f"found {eligible_count}. Check candidate metrics."
            )

    return scenarios


def solve_mclp(demand, candidates, D, p, scenario):
    weights = demand["demand_weight"].to_numpy(dtype=float)
    candidate_ids = candidates["candidate_id"].tolist()

    n_demand = len(weights)
    n_candidates = len(candidate_ids)

    model = pulp.LpProblem(
        f"MCLP_{scenario}_P{p}",
        pulp.LpMaximize
    )

    y = {
        j: pulp.LpVariable(f"y_{j}", cat="Binary")
        for j in range(n_candidates)
    }

    z = {
        i: pulp.LpVariable(f"z_{i}", cat="Binary")
        for i in range(n_demand)
    }

    model += pulp.lpSum(
        weights[i] * z[i] for i in range(n_demand)
    )

    model += pulp.lpSum(y.values()) == p

    for i in range(n_demand):
        reachable = np.flatnonzero(
            np.isfinite(D[i]) & (D[i] <= RADIUS_KM)
        )
        if len(reachable) == 0:
            model += z[i] == 0
        else:
            model += z[i] <= pulp.lpSum(y[j] for j in reachable)

    # Minimum straight-line separation between selected facilities.
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

    rows, cols = np.where(
        np.triu(distances < MIN_SEPARATION_KM, k=1)
    )

    for j, k in zip(rows, cols):
        model += y[j] + y[k] <= 1

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=TIME_LIMIT,
        gapRel=RELATIVE_GAP
    )

    model.solve(solver)
    status = pulp.LpStatus[model.status]

    result = {
        "scenario": scenario,
        "p": p,
        "coverage_radius_km": RADIUS_KM,
        "minimum_separation_km": MIN_SEPARATION_KM,
        "eligible_candidates": n_candidates,
        "solver_status": status,
        "covered_proxy_weight": np.nan,
        "total_proxy_weight": float(weights.sum()),
        "coverage_percentage": np.nan,
        "selected_facilities": 0,
    }

    selected_records = []

    if status == "Optimal":
        selected_indices = [
            j for j in range(n_candidates)
            if y[j].value() is not None and y[j].value() > 0.5
        ]

        objective = float(pulp.value(model.objective))

        result["covered_proxy_weight"] = objective
        result["coverage_percentage"] = (
            objective / weights.sum() * 100
        )
        result["selected_facilities"] = len(selected_indices)

        for j in selected_indices:
            row = candidates.iloc[j]

            selected_records.append({
                "scenario": scenario,
                "p": p,
                "candidate_id": candidate_ids[j],
                "node_id": row["node_id"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "mean_snap_distance_m": row["mean_snap_distance_m"],
                "max_snap_distance_m": row["max_snap_distance_m"],
                "weak_component_size": row["weak_component_size"],
                "strong_component_size": row["strong_component_size"],
                "in_degree": row["in_degree"],
                "out_degree": row["out_degree"],
            })

        print(
            f"  {scenario}, P={p}: {status}, "
            f"coverage={objective:.0f}/{weights.sum():.0f} "
            f"({result['coverage_percentage']:.3f}%)"
        )
    else:
        print(f"  {scenario}, P={p}: {status}")

    return result, selected_records


def main():
    print("Loading demand, candidates, distances and road metrics...")

    demand = pd.read_csv(DEMAND_PATH)

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"candidate_id": str}
    )

    metrics = pd.read_csv(
        METRICS_PATH,
        dtype={"candidate_id": str}
    )

    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)
    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    candidates = candidates.merge(
        metrics,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_road")
    )

    required_metrics = [
        "mean_snap_distance_m",
        "max_snap_distance_m",
        "weak_component_size",
        "strong_component_size",
        "in_degree",
        "out_degree",
    ]

    if candidates[required_metrics].isna().any().any():
        raise ValueError("Missing candidate road accessibility metrics.")

    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found.")

    scenarios = build_scenarios(candidates)

    weights = demand["demand_weight"].to_numpy(dtype=float)

    results = []
    selections = []

    print("\nDemand nodes:", len(demand))
    print("Total OSM-based demand proxy weight:", weights.sum())

    for scenario, excluded_ids in scenarios.items():
        eligible = candidates[
            ~candidates["candidate_id"].isin(excluded_ids)
        ].copy().reset_index(drop=True)

        candidate_ids = eligible["candidate_id"].tolist()

        missing = set(candidate_ids) - set(distance_df.columns)
        if missing:
            raise ValueError(
                f"Missing distance columns: {sorted(missing)}"
            )

        D = distance_df[candidate_ids].to_numpy(dtype=float)

        if D.shape[0] != len(demand):
            raise ValueError("Demand and distance rows do not align.")

        print(
            f"\nScenario: {scenario} "
            f"({len(eligible)} eligible candidates)"
        )

        for p in P_VALUES:
            if len(eligible) < p:
                raise ValueError("Not enough eligible candidates.")

            result, selected = solve_mclp(
                demand,
                eligible,
                D,
                p,
                scenario
            )

            results.append(result)
            selections.extend(selected)

    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)

    results_path = OUTPUT_DIR / "mclp_multiobjective_candidate_scenarios.csv"
    selections_path = OUTPUT_DIR / "mclp_multiobjective_candidate_selections.csv"

    results_df.to_csv(results_path, index=False)
    selections_df.to_csv(selections_path, index=False)

    print("\n========== SCENARIO COMPARISON ==========")
    print(
        results_df[
            [
                "scenario",
                "p",
                "eligible_candidates",
                "solver_status",
                "covered_proxy_weight",
                "coverage_percentage",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", results_path)
    print("Saved:", selections_path)


if __name__ == "__main__":
    main()
