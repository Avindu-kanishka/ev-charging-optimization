from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEMAND_PATH = ROOT / "data/processed/demand/road_demand_nodes.csv"
ROAD_PATH = ROOT / "data/processed/distance/demand_candidate_distances_km.csv"
EUCLIDEAN_PATH = ROOT / "data/processed/distance/euclidean_distances_km.csv"

ROAD_SELECTED_PATH = ROOT / "outputs/optimization/selected_locations.csv"
EUCLIDEAN_SELECTED_PATH = ROOT / "outputs/optimization/euclidean/selected_locations.csv"

ROAD_MCLP_PATH = ROOT / "outputs/optimization/mclp_results.csv"
EUCLIDEAN_MCLP_PATH = ROOT / "outputs/optimization/euclidean/mclp_results.csv"

OUTPUT_DIR = ROOT / "outputs/analysis"


def load_inputs():
    demand = pd.read_csv(DEMAND_PATH)
    demand["node_id"] = demand["node_id"].astype(str)
    demand = demand.set_index("node_id")

    road = pd.read_csv(ROAD_PATH, index_col=0)
    euclidean = pd.read_csv(EUCLIDEAN_PATH, index_col=0)

    for matrix in [road, euclidean]:
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)

    road_selected = pd.read_csv(ROAD_SELECTED_PATH)
    euclidean_selected = pd.read_csv(EUCLIDEAN_SELECTED_PATH)

    return demand, road, euclidean, road_selected, euclidean_selected


def get_selection(df, scenario="P=10"):
    selected = df.loc[
        df["scenario"].astype(str) == scenario,
        "candidate_id"
    ].astype(str).tolist()

    if not selected:
        raise ValueError(f"No facilities found for {scenario}")

    return selected


def evaluate_selection(matrix, demand, selected, selection_name, metric):
    common_ids = matrix.index.intersection(demand.index)

    aligned = matrix.loc[common_ids, selected]
    weights = demand.loc[common_ids, "demand_weight"].to_numpy(float)

    distances = aligned.to_numpy(dtype=float)
    reachable = np.isfinite(distances).any(axis=1)

    nearest = np.full(len(common_ids), np.nan)
    nearest[reachable] = np.min(distances[reachable], axis=1)

    reachable_weight = weights[reachable].sum()
    objective = np.sum(weights[reachable] * nearest[reachable])

    return {
        "facility_selection": selection_name,
        "distance_metric": metric,
        "facility_count": len(selected),
        "weighted_distance_objective": objective,
        "average_weighted_distance_km": (
            objective / reachable_weight
            if reachable_weight > 0 else np.nan
        ),
        "reachable_demand_nodes": int(reachable.sum()),
        "unreachable_demand_nodes": int((~reachable).sum()),
        "reachable_demand_weight": reachable_weight,
        "total_demand_nodes": len(common_ids),
        "total_demand_weight": weights.sum()
    }


def build_pmedian_comparison(
    demand, road, euclidean, road_selected, euclidean_selected
):
    road_ids = get_selection(road_selected)
    euclidean_ids = get_selection(euclidean_selected)

    # Best-known feasible Euclidean solution found through
    # independent evaluation of the road-selected facilities.
    best_known_euclidean_ids = road_ids

    # New diagnostic CBC solution.
    diagnostic_euclidean_ids = [
        "C0053", "C0054", "C0064", "C0068", "C0118",
        "C0130", "C0150", "C0242", "C0244", "C0254"
    ]

    selections = {
        "Road model selected": road_ids,
        "Original Euclidean selected": euclidean_ids,
        "Diagnostic Euclidean selected": diagnostic_euclidean_ids,
        "Best-known feasible Euclidean": best_known_euclidean_ids
    }

    results = []

    for name, selected in selections.items():
        for metric, matrix in [
            ("Road network", road),
            ("Euclidean", euclidean)
        ]:
            results.append(
                evaluate_selection(
                    matrix, demand, selected, name, metric
                )
            )

    return pd.DataFrame(results)


def build_mclp_comparison():
    road = pd.read_csv(ROAD_MCLP_PATH)
    euclidean = pd.read_csv(EUCLIDEAN_MCLP_PATH)

    road["distance_model"] = "Road network"
    euclidean["distance_model"] = "Euclidean"

    combined = pd.concat([road, euclidean], ignore_index=True)

    columns = [
        "distance_model",
        "scenario",
        "p",
        "coverage_radius_km",
        "solver_status",
        "covered_demand_weight",
        "total_demand_weight",
        "coverage_percentage",
        "covered_demand_nodes",
        "total_demand_nodes"
    ]

    return combined[columns].sort_values(
        ["p", "coverage_radius_km", "distance_model"]
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demand, road, euclidean, road_selected, euclidean_selected = (
        load_inputs()
    )

    pmedian = build_pmedian_comparison(
        demand, road, euclidean, road_selected, euclidean_selected
    )

    mclp = build_mclp_comparison()

    pmedian_path = OUTPUT_DIR / "comparative_pmedian_cross_evaluation.csv"
    mclp_path = OUTPUT_DIR / "comparative_mclp_evaluation.csv"

    pmedian.to_csv(pmedian_path, index=False)
    mclp.to_csv(mclp_path, index=False)

    print("\nP-MEDIAN CROSS-EVALUATION")
    print(pmedian.round(4).to_string(index=False))

    print("\nMCLP COMPARISON")
    print(mclp.round(4).to_string(index=False))

    print("\nOUTPUT FILES")
    print(pmedian_path)
    print(mclp_path)

    print("\nCOMPARISON COMPLETED")


if __name__ == "__main__":
    main()
