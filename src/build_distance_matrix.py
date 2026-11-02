
from pathlib import Path

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_road_network.graphml"
)

DEMAND_PATH = (
    BASE_DIR / "data" / "processed" / "demand"
    / "road_demand_nodes.csv"
)

CANDIDATE_PATH = (
    BASE_DIR / "data" / "processed" / "candidates"
    / "candidate_locations.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "distance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 65)
    print("DEMAND-TO-CANDIDATE SHORTEST PATH DISTANCE MATRIX")
    print("=" * 65)

    # 1. Load data
    print("\n[1/4] Loading graph and location data...")

    G = ox.load_graphml(GRAPH_PATH)

    demand = pd.read_csv(DEMAND_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)

    # Match node ID types with graph node IDs
    graph_nodes = set(G.nodes)

    def match_node_id(value):
        if value in graph_nodes:
            return value

        try:
            integer_value = int(value)
            if integer_value in graph_nodes:
                return integer_value
        except (ValueError, TypeError):
            pass

        string_value = str(value)
        if string_value in graph_nodes:
            return string_value

        raise ValueError(f"Node not found in graph: {value}")

    demand["node_id"] = demand["node_id"].apply(match_node_id)
    candidates["node_id"] = candidates["node_id"].apply(match_node_id)

    demand_nodes = demand["node_id"].tolist()
    candidate_nodes = candidates["node_id"].tolist()

    print("Demand nodes:", len(demand_nodes))
    print("Candidate nodes:", len(candidate_nodes))

    # 2. Calculate directed shortest path distances
    print("\n[2/4] Calculating Dijkstra shortest paths...")
    print("Direction: Demand node -> Candidate node")

    matrix = np.full(
        (len(demand_nodes), len(candidate_nodes)),
        np.inf,
        dtype=float
    )

    candidate_positions = {
        node: index
        for index, node in enumerate(candidate_nodes)
    }

    for i, source in enumerate(demand_nodes):

        distances = nx.single_source_dijkstra_path_length(
            G,
            source=source,
            weight="length"
        )

        for target, distance in distances.items():

            j = candidate_positions.get(target)

            if j is not None:
                matrix[i, j] = distance / 1000.0

        if (i + 1) % 100 == 0 or i == len(demand_nodes) - 1:
            print(f"Processed {i + 1}/{len(demand_nodes)} demand nodes")

    # 3. Create labelled matrix
    print("\n[3/4] Creating distance matrix...")

    matrix_df = pd.DataFrame(
        matrix,
        index=demand["node_id"].astype(str),
        columns=candidates["candidate_id"].astype(str)
    )

    matrix_df.index.name = "demand_node_id"

    matrix_path = OUTPUT_DIR / "demand_candidate_distances_km.csv"
    matrix_df.to_csv(matrix_path)

    # 4. Summary and quality checks
    print("\n[4/4] Analysing matrix...")

    finite_values = matrix[np.isfinite(matrix)]

    unreachable_pairs = int(np.isinf(matrix).sum())
    total_pairs = matrix.size

    reachable_candidates = np.isfinite(matrix).any(axis=0).sum()
    reachable_demands = np.isfinite(matrix).any(axis=1).sum()

    print("\nDISTANCE MATRIX SUMMARY")
    print("-" * 45)

    print("Matrix shape:", matrix.shape)
    print("Total demand-candidate pairs:", total_pairs)
    print("Reachable pairs:", len(finite_values))
    print("Unreachable pairs:", unreachable_pairs)

    print(
        "Reachable demand nodes:",
        reachable_demands, "/", len(demand_nodes)
    )

    print(
        "Candidates reachable from demand nodes:",
        reachable_candidates, "/", len(candidate_nodes)
    )

    if len(finite_values) > 0:
        print(
            "Minimum distance:",
            round(finite_values.min(), 3), "km"
        )

        print(
            "Median distance:",
            round(np.median(finite_values), 3), "km"
        )

        print(
            "Mean distance:",
            round(finite_values.mean(), 3), "km"
        )

        print(
            "Maximum distance:",
            round(finite_values.max(), 3), "km"
        )

    print("\nSaved:", matrix_path)
    print("\nDistance matrix generation completed.")


if __name__ == "__main__":
    main()
