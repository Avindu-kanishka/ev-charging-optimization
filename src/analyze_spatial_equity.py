
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEMAND_FILE = (
    ROOT
    / "data"
    / "processed"
    / "demand"
    / "non_candidate_demand_nodes.csv"
)

DISTANCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "distance"
    / "non_candidate_demand_candidate_distances_km.csv"
)

SELECTION_FILE = (
    ROOT
    / "outputs"
    / "optimization"
    / "mclp_lexicographic_selected_facilities.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "analysis"

OUTPUT_FILE = OUTPUT_DIR / "spatial_equity_distance_metrics.csv"


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    ("baseline", 5),
    ("baseline", 10),
    ("combined_quality", 5),
    ("combined_quality", 10),
]

COVERAGE_RADII_KM = [1, 2, 3]


# ============================================================
# WEIGHTED GINI COEFFICIENT
# ============================================================

def weighted_gini(values, weights):
    """
    Calculate weighted Gini coefficient using the
    weighted Lorenz curve.

    Only finite, non-negative distances with positive
    demand weights are included.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & (values >= 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    total_weight = weights.sum()
    total_distance = np.sum(values * weights)

    if total_weight <= 0:
        return np.nan

    if total_distance <= 0:
        return 0.0

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative_weight = np.cumsum(weights) / total_weight

    cumulative_distance = (
        np.cumsum(values * weights) / total_distance
    )

    cumulative_weight = np.insert(
        cumulative_weight, 0, 0
    )

    cumulative_distance = np.insert(
        cumulative_distance, 0, 0
    )

    area = np.sum(
        (
            cumulative_distance[1:]
            + cumulative_distance[:-1]
        )
        * np.diff(cumulative_weight)
        / 2
    )

    gini = 1 - 2 * area

    return float(np.clip(gini, 0, 1))


# ============================================================
# WEIGHTED PERCENTILE
# ============================================================

def weighted_percentile(values, weights, percentile):
    """
    Calculate a weighted percentile using cumulative
    demand weights.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative_weights = np.cumsum(weights)

    target = percentile / 100 * cumulative_weights[-1]

    index = np.searchsorted(
        cumulative_weights,
        target,
        side="left"
    )

    index = min(index, len(values) - 1)

    return float(values[index])


# ============================================================
# LOAD AND VALIDATE INPUT DATA
# ============================================================

def load_data():

    print("\n========== LOADING PROJECT DATA ==========")

    for file_path in [
        DEMAND_FILE,
        DISTANCE_FILE,
        SELECTION_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

        print(f"Found: {file_path}")

    demand_df = pd.read_csv(DEMAND_FILE)

    distance_df = pd.read_csv(
        DISTANCE_FILE,
        index_col=0
    )

    selection_df = pd.read_csv(SELECTION_FILE)

    required_demand_columns = {
        "node_id",
        "demand_weight",
    }

    required_selection_columns = {
        "scenario",
        "p",
        "candidate_id",
    }

    missing_demand = (
        required_demand_columns
        - set(demand_df.columns)
    )

    missing_selection = (
        required_selection_columns
        - set(selection_df.columns)
    )

    if missing_demand:
        raise ValueError(
            f"Missing demand columns: {missing_demand}"
        )

    if missing_selection:
        raise ValueError(
            f"Missing selection columns: {missing_selection}"
        )

    demand_df["node_id"] = (
        demand_df["node_id"].astype(str)
    )

    demand_df["demand_weight"] = pd.to_numeric(
        demand_df["demand_weight"],
        errors="coerce"
    )

    if demand_df["demand_weight"].isna().any():
        raise ValueError(
            "Demand weights contain missing or invalid values."
        )

    if (demand_df["demand_weight"] < 0).any():
        raise ValueError(
            "Demand weights cannot be negative."
        )

    if demand_df["node_id"].duplicated().any():
        raise ValueError(
            "Duplicate demand node IDs found."
        )

    distance_df.index = distance_df.index.astype(str)
    distance_df.columns = distance_df.columns.astype(str)

    distance_df = distance_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Negative distances are invalid.
    distance_df = distance_df.where(
        distance_df >= 0
    )

    selection_df["scenario"] = (
        selection_df["scenario"].astype(str)
    )

    selection_df["candidate_id"] = (
        selection_df["candidate_id"].astype(str)
    )

    selection_df["p"] = pd.to_numeric(
        selection_df["p"],
        errors="raise"
    ).astype(int)

    # Align distance matrix rows with demand records.
    missing_nodes = set(demand_df["node_id"]) - set(
        distance_df.index
    )

    if missing_nodes:
        raise ValueError(
            f"{len(missing_nodes)} demand nodes are missing "
            "from the distance matrix. Example: "
            f"{list(missing_nodes)[:10]}"
        )

    distance_df = distance_df.loc[
        demand_df["node_id"]
    ]

    print(f"\nDemand nodes: {len(demand_df)}")
    print(
        "Total OSM proxy weight:",
        demand_df["demand_weight"].sum()
    )

    print(
        "Distance matrix shape:",
        distance_df.shape
    )

    print(
        "Facility selection records:",
        len(selection_df)
    )

    return demand_df, distance_df, selection_df


# ============================================================
# ANALYZE ONE CONFIGURATION
# ============================================================

def analyze_configuration(
    scenario,
    p,
    demand_df,
    distance_df,
    selection_df
):

    label = f"{scenario}, P={p}"

    print(f"\n========== {label} ==========")

    selected_rows = selection_df[
        (selection_df["scenario"] == scenario)
        & (selection_df["p"] == p)
    ]

    if selected_rows.empty:
        raise ValueError(
            f"No selected facilities found for {label}"
        )

    selected_candidates = (
        selected_rows["candidate_id"]
        .drop_duplicates()
        .tolist()
    )

    if len(selected_candidates) != p:
        raise ValueError(
            f"{label}: expected {p} unique facilities, "
            f"but found {len(selected_candidates)}."
        )

    missing_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate not in distance_df.columns
    ]

    if missing_candidates:
        raise ValueError(
            f"{label}: selected facilities are missing "
            f"from the distance matrix: {missing_candidates}"
        )

    print("Selected facilities:", selected_candidates)

    # Road-network distances from each demand node
    # to the selected facilities.
    selected_distances = distance_df[
        selected_candidates
    ].to_numpy(dtype=float)

    # A demand node is reachable if at least one
    # selected facility has a finite road distance.
    reachable_mask = np.isfinite(
        selected_distances
    ).any(axis=1)

    # Nearest selected facility distance.
    nearest_distances = np.full(
        len(demand_df),
        np.inf,
        dtype=float
    )

    if reachable_mask.any():

        nearest_distances[reachable_mask] = np.min(
            selected_distances[reachable_mask],
            axis=1
        )

    weights = demand_df[
        "demand_weight"
    ].to_numpy(dtype=float)

    total_nodes = len(demand_df)

    total_proxy_weight = weights.sum()

    unreachable_mask = ~reachable_mask

    reachable_distances = nearest_distances[
        reachable_mask
    ]

    reachable_weights = weights[
        reachable_mask
    ]

    unreachable_nodes = int(
        unreachable_mask.sum()
    )

    unreachable_proxy_weight = float(
        weights[unreachable_mask].sum()
    )

    reachable_proxy_weight = float(
        reachable_weights.sum()
    )

    print(f"Total demand nodes: {total_nodes}")

    print(
        f"Reachable demand nodes: "
        f"{reachable_mask.sum()}"
    )

    print(
        f"Unreachable demand nodes: "
        f"{unreachable_nodes}"
    )

    print(
        f"Unreachable OSM proxy weight: "
        f"{unreachable_proxy_weight:.0f}"
    )

    # Initialize metrics.
    result = {
        "scenario": scenario,
        "p": p,
        "selected_facilities": ";".join(
            selected_candidates
        ),
        "total_demand_nodes": total_nodes,
        "reachable_demand_nodes": int(
            reachable_mask.sum()
        ),
        "unreachable_demand_nodes": unreachable_nodes,
        "reachable_demand_nodes_pct": (
            100 * reachable_mask.sum() / total_nodes
            if total_nodes else np.nan
        ),
        "total_proxy_weight": total_proxy_weight,
        "reachable_proxy_weight": reachable_proxy_weight,
        "unreachable_proxy_weight": unreachable_proxy_weight,
        "unreachable_proxy_weight_pct": (
            100 * unreachable_proxy_weight
            / total_proxy_weight
            if total_proxy_weight > 0 else np.nan
        ),
        "mean_distance_km": np.nan,
        "median_distance_km": np.nan,
        "weighted_mean_distance_km": np.nan,
        "p90_distance_km": np.nan,
        "p95_distance_km": np.nan,
        "weighted_p90_distance_km": np.nan,
        "weighted_p95_distance_km": np.nan,
        "maximum_distance_km": np.nan,
        "distance_gini": np.nan,
    }

    # Distance inequality metrics are calculated only
    # among reachable nodes. Unreachable demand is
    # reported separately above.
    if len(reachable_distances) > 0:

        result["mean_distance_km"] = float(
            np.mean(reachable_distances)
        )

        result["median_distance_km"] = float(
            np.median(reachable_distances)
        )

        result["weighted_mean_distance_km"] = float(
            np.average(
                reachable_distances,
                weights=reachable_weights
            )
        )

        result["p90_distance_km"] = float(
            np.percentile(
                reachable_distances,
                90
            )
        )

        result["p95_distance_km"] = float(
            np.percentile(
                reachable_distances,
                95
            )
        )

        result["weighted_p90_distance_km"] = (
            weighted_percentile(
                reachable_distances,
                reachable_weights,
                90
            )
        )

        result["weighted_p95_distance_km"] = (
            weighted_percentile(
                reachable_distances,
                reachable_weights,
                95
            )
        )

        result["maximum_distance_km"] = float(
            np.max(reachable_distances)
        )

        result["distance_gini"] = weighted_gini(
            reachable_distances,
            reachable_weights
        )

    # Calculate demand beyond each access threshold.
    # Unreachable demand is counted as beyond every
    # finite threshold.
    for radius in COVERAGE_RADII_KM:

        beyond_radius = (
            (~reachable_mask)
            | (nearest_distances > radius)
        )

        beyond_nodes = int(
            beyond_radius.sum()
        )

        beyond_weight = float(
            weights[beyond_radius].sum()
        )

        result[
            f"demand_nodes_beyond_{radius}km"
        ] = beyond_nodes

        result[
            f"demand_nodes_beyond_{radius}km_pct"
        ] = (
            100 * beyond_nodes / total_nodes
            if total_nodes > 0 else np.nan
        )

        result[
            f"proxy_weight_beyond_{radius}km"
        ] = beyond_weight

        result[
            f"proxy_weight_beyond_{radius}km_pct"
        ] = (
            100 * beyond_weight / total_proxy_weight
            if total_proxy_weight > 0 else np.nan
        )

    print(
        "Weighted mean distance:",
        f"{result['weighted_mean_distance_km']:.3f} km"
        if np.isfinite(result["weighted_mean_distance_km"])
        else "N/A"
    )

    print(
        "P90 distance:",
        f"{result['p90_distance_km']:.3f} km"
        if np.isfinite(result["p90_distance_km"])
        else "N/A"
    )

    print(
        "Weighted Gini:",
        f"{result['distance_gini']:.4f}"
        if np.isfinite(result["distance_gini"])
        else "N/A"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    demand_df, distance_df, selection_df = load_data()

    results = []

    for scenario, p in CONFIGURATIONS:

        result = analyze_configuration(
            scenario=scenario,
            p=p,
            demand_df=demand_df,
            distance_df=distance_df,
            selection_df=selection_df
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Display useful metrics.
    display_columns = [
        "scenario",
        "p",
        "reachable_demand_nodes",
        "unreachable_demand_nodes",
        "unreachable_proxy_weight",
        "mean_distance_km",
        "weighted_mean_distance_km",
        "p90_distance_km",
        "p95_distance_km",
        "weighted_p90_distance_km",
        "weighted_p95_distance_km",
        "maximum_distance_km",
        "distance_gini",
        "proxy_weight_beyond_1km_pct",
        "proxy_weight_beyond_2km_pct",
        "proxy_weight_beyond_3km_pct",
    ]

    print("\n========== SPATIAL EQUITY METRICS ==========\n")

    print(
        results_df[display_columns].to_string(
            index=False
        )
    )

    print("\nSaved results:")
    print(OUTPUT_FILE)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
