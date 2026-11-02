
from pathlib import Path

import numpy as np
import pandas as pd
import pulp


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMAND_PATH = (
    PROJECT_ROOT / "data/processed/demand/road_demand_nodes.csv"
)

CANDIDATE_PATH = (
    PROJECT_ROOT / "data/processed/candidates/candidate_locations.csv"
)

DISTANCE_PATH = (
    PROJECT_ROOT
    / "data/processed/distance/euclidean_distances_km.csv"
)


OUTPUT_DIR = PROJECT_ROOT / "outputs/optimization/euclidean"

P_VALUES = [5, 10, 15]
COVERAGE_RADII = [3, 5, 10]

TIME_LIMIT = 300
RELATIVE_GAP = 0.01


# ============================================================
# 1. LOAD AND ALIGN DATA
# ============================================================

def load_data():

    print("\n[1/4] Loading optimization data...")

    demand = pd.read_csv(DEMAND_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)
    distances = pd.read_csv(DISTANCE_PATH, index_col=0)

    # Normalize IDs
    demand["node_id"] = demand["node_id"].astype(str)
    candidates["candidate_id"] = (
        candidates["candidate_id"].astype(str)
    )

    distances.index = distances.index.astype(str)
    distances.columns = distances.columns.astype(str)

    # Align demand rows
    demand = demand.set_index("node_id")

    missing_demand = distances.index.difference(demand.index)

    if len(missing_demand):
        raise ValueError(
            f"Missing demand nodes: {len(missing_demand)}"
        )

    demand = demand.loc[distances.index]

    # Align candidate columns
    candidates = candidates.set_index("candidate_id")

    missing_candidates = distances.columns.difference(
        candidates.index
    )

    if len(missing_candidates):
        raise ValueError(
            f"Missing candidates: {len(missing_candidates)}"
        )

    candidates = candidates.loc[distances.columns]

    D = distances.to_numpy(dtype=float)

    weights = demand["demand_weight"].to_numpy(dtype=float)

    demand_ids = distances.index.tolist()
    candidate_ids = distances.columns.tolist()

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Invalid demand weights detected.")

    print(f"   Demand nodes: {len(demand_ids):,}")
    print(f"   Candidate locations: {len(candidate_ids):,}")
    print(f"   Distance matrix: {D.shape}")
    print(f"   Unreachable pairs: {(~np.isfinite(D)).sum():,}")
    print(f"   Total demand proxy weight: {weights.sum():,.0f}")

    return D, weights, demand_ids, candidate_ids, candidates


# ============================================================
# 2. SOLVER CONFIGURATION
# ============================================================

def create_solver():

    return pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=TIME_LIMIT,
        gapRel=RELATIVE_GAP
    )


# ============================================================
# 3. SAVE SELECTED LOCATIONS
# ============================================================

def get_selected_locations(
    model,
    y,
    candidate_ids,
    candidates,
    method,
    scenario
):

    selected = []

    for j, candidate_id in enumerate(candidate_ids):

        value = pulp.value(y[j])

        if value is not None and value > 0.5:

            row = candidates.loc[candidate_id]

            selected.append({
                "method": method,
                "scenario": scenario,
                "candidate_id": candidate_id,
                "node_id": row["node_id"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "candidate_sources": row["candidate_sources"],
                "mean_snap_distance_m": row[
                    "mean_snap_distance_m"
                ],
                "max_snap_distance_m": row[
                    "max_snap_distance_m"
                ]
            })

    return selected


# ============================================================
# 4. P-MEDIAN OPTIMIZATION
# ============================================================

def solve_p_median(
    D,
    weights,
    demand_ids,
    candidate_ids,
    candidates
):

    print("\n[2/4] Running P-Median optimization...")

    results = []
    selected_locations = []

    n_demand, n_candidates = D.shape

    finite = np.isfinite(D)

    # Demand nodes with at least one reachable candidate
    assignable_rows = np.where(finite.any(axis=1))[0]

    # Demand nodes with no reachable candidate
    unassignable_rows = np.where(~finite.any(axis=1))[0]

    assignable_weight = weights[assignable_rows].sum()
    unassignable_weight = weights[unassignable_rows].sum()

    print(f"   Assignable demand nodes: {len(assignable_rows):,}")
    print(f"   Unassignable demand nodes: {len(unassignable_rows):,}")

    print(
        f"   Unassignable proxy weight: "
        f"{unassignable_weight:,.0f}"
    )

    for p in P_VALUES:

        print(f"\n   Solving P-Median for P = {p}...")

        model = pulp.LpProblem(
            f"P_Median_{p}",
            pulp.LpMinimize
        )

        # Facility selection variables
        y = {
            j: pulp.LpVariable(
                f"facility_{j}",
                cat=pulp.LpBinary
            )
            for j in range(n_candidates)
        }

        # Assignment variables only for reachable pairs
        x = {}

        for i in assignable_rows:

            for j in np.where(finite[i])[0]:

                x[(i, j)] = pulp.LpVariable(
                    f"x_{i}_{j}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpContinuous
                )

        # Objective: minimize weighted road distance
        model += pulp.lpSum(
            weights[i] * D[i, j] * x[(i, j)]
            for (i, j) in x
        )

        # Select exactly P facilities
        model += pulp.lpSum(y[j] for j in range(n_candidates)) == p

        # Every assignable demand node must be assigned once
        for i in assignable_rows:

            model += pulp.lpSum(
                x[(i, j)]
                for j in np.where(finite[i])[0]
            ) == 1

        # Assignment is possible only to an open facility
        for (i, j), variable in x.items():

            model += variable <= y[j]

        # Solve
        model.solve(create_solver())

        status = pulp.LpStatus[model.status]

        objective = pulp.value(model.objective)

        print(f"   Solver status: {status}")

        if status not in ("Optimal", "Not Solved", "Integer Feasible"):
            print("   No usable solution for this scenario.")
            continue

        if objective is None:
            print("   Objective value unavailable.")
            continue

        # Save selected locations
        selected = get_selected_locations(
            model,
            y,
            candidate_ids,
            candidates,
            "P-Median",
            f"P={p}"
        )

        selected_locations.extend(selected)

        if len(selected) != p:
            print("   Warning: selected facility count differs from P.")

        # Average weighted distance
        average_distance = (
            objective / assignable_weight
            if assignable_weight > 0
            else np.nan
        )

        results.append({
            "method": "P-Median",
            "scenario": f"P={p}",
            "p": p,
            "solver_status": status,
            "objective_weighted_distance": objective,
            "average_weighted_distance_km": average_distance,
            "assignable_demand_nodes": len(assignable_rows),
            "unassignable_demand_nodes": len(unassignable_rows),
            "assignable_demand_weight": assignable_weight,
            "unassignable_demand_weight": unassignable_weight,
            "selected_facilities": len(selected)
        })

        print(f"   Selected facilities: {len(selected)}")
        print(f"   Weighted distance objective: {objective:.3f}")
        print(f"   Average weighted distance: {average_distance:.3f} km")

    return results, selected_locations


# ============================================================
# 5. MAXIMUM COVERAGE LOCATION PROBLEM
# ============================================================


def solve_mclp(
    D,
    weights,
    demand_ids,
    candidate_ids,
    candidates
):

    print("\n[3/4] Running MCLP optimization...")

    results = []
    selected_locations = []

    n_demand, n_candidates = D.shape
    total_demand_weight = weights.sum()

    for p in P_VALUES:

        for radius in COVERAGE_RADII:

            print(
                f"\n   Solving MCLP: P={p}, "
                f"Radius={radius} km..."
            )

            model = pulp.LpProblem(
                f"MCLP_P{p}_R{radius}",
                pulp.LpMaximize
            )

            # Facility selection
            y = {
                j: pulp.LpVariable(
                    f"facility_{j}",
                    cat=pulp.LpBinary
                )
                for j in range(n_candidates)
            }

            # Demand coverage
            z = {
                i: pulp.LpVariable(
                    f"covered_{i}",
                    cat=pulp.LpBinary
                )
                for i in range(n_demand)
            }

            # Objective
            model += pulp.lpSum(
                weights[i] * z[i]
                for i in range(n_demand)
            )

            # Select exactly P facilities
            model += pulp.lpSum(
                y[j] for j in range(n_candidates)
            ) == p

            # Coverage constraints
            for i in range(n_demand):

                reachable = np.where(
                    np.isfinite(D[i]) &
                    (D[i] <= radius)
                )[0]

                model += z[i] <= pulp.lpSum(
                    y[j] for j in reachable
                )

            # Solve
            model.solve(create_solver())

            status = pulp.LpStatus[model.status]

            covered_weight = pulp.value(
                model.objective
            )

            print(f"   Solver status: {status}")

            if status not in (
                "Optimal",
                "Not Solved",
                "Integer Feasible"
            ):
                print("   No usable solution.")
                continue

            if covered_weight is None:
                print("   Objective unavailable.")
                continue

            selected = get_selected_locations(
                model,
                y,
                candidate_ids,
                candidates,
                "MCLP",
                f"P={p}, Radius={radius}km"
            )

            selected_locations.extend(selected)

            covered_nodes = sum(
                1 for i in range(n_demand)
                if pulp.value(z[i]) is not None
                and pulp.value(z[i]) > 0.5
            )

            coverage = (
                covered_weight / total_demand_weight * 100
                if total_demand_weight > 0
                else 0
            )

            results.append({
                "method": "MCLP",
                "scenario": f"P={p}, Radius={radius}km",
                "p": p,
                "coverage_radius_km": radius,
                "solver_status": status,
                "covered_demand_weight": covered_weight,
                "total_demand_weight": total_demand_weight,
                "coverage_percentage": coverage,
                "covered_demand_nodes": covered_nodes,
                "total_demand_nodes": n_demand,
                "selected_facilities": len(selected)
            })

            print(
                f"   Selected facilities: {len(selected)}"
            )
            print(
                f"   Covered demand: "
                f"{covered_weight:,.0f}/{total_demand_weight:,.0f}"
            )
            print(
                f"   Coverage: {coverage:.2f}%"
            )

    return results, selected_locations


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("EUCLIDEAN DISTANCE-BASED EV CHARGING STATION OPTIMIZATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    D, weights, demand_ids, candidate_ids, candidates = load_data()

    # P-Median
    p_results, p_locations = solve_p_median(
        D,
        weights,
        demand_ids,
        candidate_ids,
        candidates
    )

    # MCLP
    m_results, m_locations = solve_mclp(
        D,
        weights,
        demand_ids,
        candidate_ids,
        candidates
    )

    print("\n[4/4] Saving optimization results...")

    # Save separate result files
    p_df = pd.DataFrame(p_results)
    m_df = pd.DataFrame(m_results)

    p_df.to_csv(
        OUTPUT_DIR / "p_median_results.csv",
        index=False
    )

    m_df.to_csv(
        OUTPUT_DIR / "mclp_results.csv",
        index=False
    )

    # Save all selected locations
    all_locations = p_locations + m_locations

    locations_df = pd.DataFrame(all_locations)

    locations_df.to_csv(
        OUTPUT_DIR / "selected_locations.csv",
        index=False
    )

    # Combined summary
    summary_df = pd.concat(
        [p_df, m_df],
        ignore_index=True,
        sort=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "optimization_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("OPTIMIZATION COMPLETED")
    print("=" * 65)

    print("\nP-MEDIAN RESULTS")
    if not p_df.empty:
        print(p_df.to_string(index=False))
    else:
        print("No P-Median results available.")

    print("\nMCLP RESULTS")
    if not m_df.empty:
        print(m_df.to_string(index=False))
    else:
        print("No MCLP results available.")

    print("\nSaved output directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- p_median_results.csv")
    print("- mclp_results.csv")
    print("- selected_locations.csv")
    print("- optimization_summary.csv")


if __name__ == "__main__":
    main()
