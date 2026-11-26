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
RADII_KM = [1.0, 2.0, 3.0, 5.0]
MIN_SEPARATION_KM = 0.5

TIME_LIMIT = 300
RELATIVE_GAP = 0.01


def haversine_matrix(candidates):
    earth_radius = 6371.0088

    lat = np.radians(candidates["latitude"].to_numpy(dtype=float))
    lon = np.radians(candidates["longitude"].to_numpy(dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin(dlon / 2) ** 2
    )

    distances = 2 * earth_radius * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    np.fill_diagonal(distances, np.inf)
    return distances


def create_weight_scenarios(original_weights):
    total = original_weights.sum()

    scenarios = {
        "original": original_weights.copy(),
        "uniform": np.ones(len(original_weights)),
        "square_root": np.sqrt(original_weights),
        "logarithmic": np.log1p(original_weights),
    }

    normalized = {}

    for name, values in scenarios.items():
        if not np.isfinite(values).all() or values.sum() <= 0:
            raise ValueError(f"Invalid weights in scenario: {name}")

        normalized[name] = values * (total / values.sum())

    return normalized


def solve_mclp(D, weights, candidate_ids, separation_matrix,
               p, radius, scenario_name):

    model = pulp.LpProblem(
        f"MCLP_{scenario_name}_P{p}_R{radius}",
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
            np.isfinite(D[i]) & (D[i] <= radius)
        )

        if len(reachable) == 0:
            model += z[i] == 0
        else:
            model += z[i] <= pulp.lpSum(y[j] for j in reachable)

    rows, cols = np.where(
        np.triu(separation_matrix < MIN_SEPARATION_KM, k=1)
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
        "weight_scenario": scenario_name,
        "p": p,
        "coverage_radius_km": radius,
        "minimum_separation_km": MIN_SEPARATION_KM,
        "solver_status": status,
        "covered_proxy_weight": np.nan,
        "coverage_percentage": np.nan,
        "selected_facilities": 0,
        "total_proxy_weight": float(weights.sum()),
    }

    selected = []

    if status == "Optimal":
        objective = float(pulp.value(model.objective))

        selected_indices = [
            j for j in range(len(candidate_ids))
            if pulp.value(y[j]) is not None
            and pulp.value(y[j]) > 0.5
        ]

        if len(selected_indices) != p:
            raise RuntimeError(
                f"Expected {p} facilities, got {len(selected_indices)}"
            )

        result["covered_proxy_weight"] = objective
        result["coverage_percentage"] = (
            objective / weights.sum() * 100
        )
        result["selected_facilities"] = len(selected_indices)

        selected = [
            candidate_ids[j] for j in selected_indices
        ]

    return result, selected


def main():
    demand = pd.read_csv(DEMAND_PATH)
    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"candidate_id": str}
    )

    distance_df = pd.read_csv(DISTANCE_PATH, index_col=0)

    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    demand_ids = demand["node_id"].astype(str).tolist()
    matrix_ids = distance_df.index.tolist()

    if demand_ids != matrix_ids:
        raise ValueError(
            "Demand node order does not match distance matrix rows."
        )

    if demand["node_id"].duplicated().any():
        raise ValueError("Duplicate demand node IDs found.")

    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found.")

    original_count = len(candidates)

    candidates = candidates[
        ~candidates["candidate_id"].isin(EXCLUDED_CANDIDATES)
    ].copy().reset_index(drop=True)

    candidate_ids = candidates["candidate_id"].tolist()

    missing = set(candidate_ids) - set(distance_df.columns)

    if missing:
        raise ValueError(
            f"Missing candidate distance columns: {sorted(missing)}"
        )

    D = distance_df[candidate_ids].to_numpy(dtype=float)

    original_weights = demand["demand_weight"].to_numpy(dtype=float)

    if not np.isfinite(original_weights).all():
        raise ValueError("Demand weights contain non-finite values.")

    if (original_weights <= 0).any():
        raise ValueError("Demand weights must be positive.")

    if not np.isfinite(D).any():
        raise ValueError("No reachable demand-candidate pairs.")

    weight_scenarios = create_weight_scenarios(original_weights)
    separation_matrix = haversine_matrix(candidates)

    print("Original candidates:", original_count)
    print("Excluded candidates:", sorted(EXCLUDED_CANDIDATES))
    print("Eligible candidates:", len(candidates))
    print("Demand nodes:", len(demand))
    print("Reachable pairs:", int(np.isfinite(D).sum()))
    print("Original total proxy weight:", original_weights.sum())
    print("Minimum separation:", MIN_SEPARATION_KM, "km")

    print("\nWeight scenario totals:")
    for name, weights in weight_scenarios.items():
        print(f"{name}: {weights.sum():.4f}")

    results = []
    selections = []

    total_scenarios = (
        len(weight_scenarios) * len(P_VALUES) * len(RADII_KM)
    )
    completed = 0

    for weight_name, weights in weight_scenarios.items():
        for p in P_VALUES:
            for radius in RADII_KM:
                completed += 1

                print(
                    f"\n[{completed}/{total_scenarios}] "
                    f"Weight={weight_name}, P={p}, "
                    f"Radius={radius} km"
                )

                result, selected = solve_mclp(
                    D=D,
                    weights=weights,
                    candidate_ids=candidate_ids,
                    separation_matrix=separation_matrix,
                    p=p,
                    radius=radius,
                    scenario_name=weight_name
                )

                results.append(result)

                print("Status:", result["solver_status"])

                if result["solver_status"] == "Optimal":
                    print(
                        f"Coverage: "
                        f"{result['covered_proxy_weight']:.2f}/"
                        f"{result['total_proxy_weight']:.2f} "
                        f"({result['coverage_percentage']:.3f}%)"
                    )
                    print("Selected:", selected)

                    for candidate_id in selected:
                        row = candidates[
                            candidates["candidate_id"] == candidate_id
                        ].iloc[0]

                        selections.append({
                            "weight_scenario": weight_name,
                            "p": p,
                            "coverage_radius_km": radius,
                            "minimum_separation_km": MIN_SEPARATION_KM,
                            "candidate_id": candidate_id,
                            "node_id": row["node_id"],
                            "longitude": row["longitude"],
                            "latitude": row["latitude"],
                        })

                else:
                    print("Scenario not accepted as optimal.")

    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)

    results_path = (
        OUTPUT_DIR / "mclp_demand_weight_radius_sensitivity.csv"
    )
    selections_path = (
        OUTPUT_DIR / "mclp_demand_weight_radius_selected_locations.csv"
    )

    results_df.to_csv(results_path, index=False)
    selections_df.to_csv(selections_path, index=False)

    print("\n========== FINAL SUMMARY ==========")
    print("Scenarios:", len(results_df))
    print(
        "Optimal scenarios:",
        int(results_df["solver_status"].eq("Optimal").sum())
    )
    print(
        "Non-optimal scenarios:",
        int(results_df["solver_status"].ne("Optimal").sum())
    )
    print("Selected-location records:", len(selections_df))
    print("Results:", results_path)
    print("Selections:", selections_path)


if __name__ == "__main__":
    main()
