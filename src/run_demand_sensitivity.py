from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pulp


ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = ROOT / "data/raw/road_network/colombo_road_network.graphml"
SOURCE_DEMAND_PATH = ROOT / "data/raw/demand/colombo_demand_proxies.geojson"
CANDIDATE_PATH = ROOT / "data/processed/candidates/candidate_locations.csv"

OUTPUT_DEMAND = ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
OUTPUT_DISTANCE = ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"
OUTPUT_RESULTS = ROOT / "outputs/optimization/non_candidate_mclp_sensitivity.csv"

EXCLUDED_AMENITIES = {"parking", "fuel", "marketplace"}
EXCLUDED_SHOPS = {"mall", "supermarket", "shopping_mall"}
P_VALUES = [5, 10, 15]
COVERAGE_RADII_KM = [3, 5, 10]


def main() -> None:
    print("=" * 72)
    print("EVISION AI - NON-CANDIDATE POI DEMAND SENSITIVITY EXPERIMENT")
    print("=" * 72)

    for path in (GRAPH_PATH, SOURCE_DEMAND_PATH, CANDIDATE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    graph = ox.load_graphml(GRAPH_PATH)
    source = gpd.read_file(SOURCE_DEMAND_PATH).to_crs("EPSG:4326")
    candidates = pd.read_csv(CANDIDATE_PATH, dtype={"candidate_id": str})

    # Keep non-candidate POI categories as an alternative proxy scenario.
    amenity = source["amenity"] if "amenity" in source.columns else pd.Series(index=source.index, dtype="object")
    shop = source["shop"] if "shop" in source.columns else pd.Series(index=source.index, dtype="object")

    candidate_poi_mask = (
        amenity.isin(EXCLUDED_AMENITIES)
        | shop.isin(EXCLUDED_SHOPS)
    )
    alternative = source.loc[~candidate_poi_mask].copy()
    alternative = alternative[
        alternative.geometry.notna() & ~alternative.geometry.is_empty
    ].copy()

    if alternative.empty:
        raise ValueError("No non-candidate POI features remain after filtering.")

    print(f"Original POI features: {len(source):,}")
    print(f"Candidate-type POIs excluded: {int(candidate_poi_mask.sum()):,}")
    print(f"Alternative demand POIs retained: {len(alternative):,}")

    # Match retained features to the existing road graph.
    x_coords = alternative.geometry.x.to_numpy()
    y_coords = alternative.geometry.y.to_numpy()
    nearest_nodes, snap_distances = ox.distance.nearest_nodes(
        graph, X=x_coords, Y=y_coords, return_dist=True
    )
    alternative["node_id"] = nearest_nodes
    alternative["snap_distance_m"] = snap_distances
    alternative["proxy_count"] = 1

    demand = (
        alternative.groupby("node_id")
        .agg(
            demand_weight=("proxy_count", "sum"),
            proxy_count=("proxy_count", "sum"),
            mean_snap_distance_m=("snap_distance_m", "mean"),
            max_snap_distance_m=("snap_distance_m", "max"),
        )
        .reset_index()
    )

    demand["longitude"] = demand["node_id"].map(lambda node: graph.nodes[node]["x"])
    demand["latitude"] = demand["node_id"].map(lambda node: graph.nodes[node]["y"])
    demand.to_csv(OUTPUT_DEMAND, index=False)

    # Preserve candidate column order from the existing candidate dataset.
    candidates["node_id"] = candidates["node_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    graph_node_lookup = {str(node): node for node in graph.nodes}
    candidates["graph_node"] = candidates["node_id"].map(graph_node_lookup)
    if candidates["graph_node"].isna().any():
        raise ValueError("One or more candidate road nodes are absent from the graph.")

    demand_nodes = demand["node_id"].tolist()
    candidate_nodes = candidates["graph_node"].tolist()
    candidate_ids = candidates["candidate_id"].tolist()

    # Directed demand-to-candidate shortest-path distances, in kilometres.
    matrix = np.full((len(demand_nodes), len(candidate_nodes)), np.inf, dtype=float)
    candidate_positions = {node: j for j, node in enumerate(candidate_nodes)}

    for i, source_node in enumerate(demand_nodes):
        lengths = nx.single_source_dijkstra_path_length(
            graph, source=source_node, weight="length"
        )
        for target_node, metres in lengths.items():
            j = candidate_positions.get(target_node)
            if j is not None:
                matrix[i, j] = float(metres) / 1000.0

        if (i + 1) % 100 == 0 or i == len(demand_nodes) - 1:
            print(f"Distance rows processed: {i + 1:,}/{len(demand_nodes):,}")

    distance_df = pd.DataFrame(
        matrix,
        index=demand["node_id"].astype(str),
        columns=candidate_ids,
    )
    distance_df.index.name = "demand_node_id"
    distance_df.to_csv(OUTPUT_DISTANCE)

    weights = demand["demand_weight"].to_numpy(dtype=float)
    overlap = set(demand["node_id"].astype(str)) & set(candidates["node_id"])
    print(f"Alternative demand nodes: {len(demand):,}")
    print(f"Candidate locations: {len(candidates):,}")
    print(f"Shared road nodes remaining: {len(overlap):,}")
    print(f"Reachable pairs: {int(np.isfinite(matrix).sum()):,}")
    print(f"Unreachable pairs: {int((~np.isfinite(matrix)).sum()):,}")

    # Run the same MCLP formulation to compare this sensitivity scenario.
    results = []
    n_demand, n_candidates = matrix.shape
    total_weight = float(weights.sum())

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300, gapRel=0.01)

    for p in P_VALUES:
        for radius in COVERAGE_RADII_KM:
            model = pulp.LpProblem(
                f"NonCandidate_MCLP_P{p}_R{radius}",
                pulp.LpMaximize,
            )
            y = {
                j: pulp.LpVariable(f"open_{j}", cat=pulp.LpBinary)
                for j in range(n_candidates)
            }
            z = {
                i: pulp.LpVariable(f"covered_{i}", cat=pulp.LpBinary)
                for i in range(n_demand)
            }

            model += pulp.lpSum(weights[i] * z[i] for i in range(n_demand))
            model += pulp.lpSum(y[j] for j in range(n_candidates)) == p

            for i in range(n_demand):
                reachable = np.where(
                    np.isfinite(matrix[i]) & (matrix[i] <= radius)
                )[0]
                model += z[i] <= pulp.lpSum(y[j] for j in reachable)

            model.solve(solver)
            status = pulp.LpStatus[model.status]
            covered_weight = pulp.value(model.objective)

            if status == "Optimal" and covered_weight is not None:
                selected_count = sum(
                    1 for j in range(n_candidates)
                    if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5
                )
                covered_nodes = sum(
                    1 for i in range(n_demand)
                    if pulp.value(z[i]) is not None and pulp.value(z[i]) > 0.5
                )
                coverage_pct = (
                    float(covered_weight) / total_weight * 100
                    if total_weight > 0 else 0.0
                )
            else:
                selected_count = None
                covered_nodes = None
                coverage_pct = None

            results.append({
                "scenario": "non_candidate_poi_demand",
                "p": p,
                "coverage_radius_km": radius,
                "solver_status": status,
                "covered_proxy_weight": covered_weight,
                "total_proxy_weight": total_weight,
                "coverage_percentage": coverage_pct,
                "covered_demand_nodes": covered_nodes,
                "total_demand_nodes": n_demand,
                "selected_facilities": selected_count,
                "shared_candidate_demand_road_nodes": len(overlap),
            })
            print(
                f"P={p}, radius={radius} km: {status}; "
                f"proxy coverage={coverage_pct if coverage_pct is not None else 'N/A'}"
            )

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_RESULTS, index=False)

    print("\nSensitivity experiment completed.")
    print(f"Alternative demand nodes: {OUTPUT_DEMAND}")
    print(f"Alternative distance matrix: {OUTPUT_DISTANCE}")
    print(f"MCLP comparison results: {OUTPUT_RESULTS}")
    print(
        "\nInterpretation: this is a sensitivity scenario using non-candidate "
        "OSM POI categories, not observed EV charging demand."
    )


if __name__ == "__main__":
    main()
