
from pathlib import Path

import osmnx as ox
import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_road_network.graphml"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("ROAD NETWORK GRAPH ANALYSIS")
    print("=" * 60)

    # 1. Load graph
    print("\n[1/4] Loading road network...")

    G = ox.load_graphml(GRAPH_PATH)

    print("Graph type:", type(G).__name__)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # 2. Analyse connected components
    print("\n[2/4] Analysing connected components...")

    weak_components = sorted(
        nx.weakly_connected_components(G),
        key=len,
        reverse=True
    )

    strong_components = sorted(
        nx.strongly_connected_components(G),
        key=len,
        reverse=True
    )

    component_rows = []

    for i, component in enumerate(weak_components, start=1):
        component_rows.append({
            "component_id": i,
            "nodes": len(component),
            "percentage": round(
                len(component) / G.number_of_nodes() * 100, 2
            )
        })

    components_df = pd.DataFrame(component_rows)

    print("\nWeakly connected components:")
    print(components_df.to_string(index=False))

    print("\nNumber of strongly connected components:",
          len(strong_components))

    print("Largest strongly connected component:",
          len(strong_components[0]))

    components_df.to_csv(
        OUTPUT_DIR / "weak_components.csv",
        index=False
    )

    # 3. Calculate road lengths
    print("\n[3/4] Calculating road lengths...")

    directed_length = sum(
        float(data["length"])
        for _, _, _, data in G.edges(keys=True, data=True)
    )

    undirected_graph = ox.convert.to_undirected(G)

    undirected_length = sum(
        float(data["length"])
        for _, _, _, data in undirected_graph.edges(
            keys=True, data=True
        )
    )

    print(f"Directed edge-length sum: {directed_length / 1000:.2f} km")
    print(f"Undirected road-length sum: {undirected_length / 1000:.2f} km")

    # 4. Test Dijkstra on largest strongly connected component
    print("\n[4/4] Testing Dijkstra shortest paths...")

    largest_scc = strong_components[0]
    H = G.subgraph(largest_scc).copy()

    # Select three geographically ordered nodes
    selected_nodes = sorted(
        H.nodes,
        key=lambda n: (H.nodes[n]["x"], H.nodes[n]["y"])
    )

    positions = [
        0,
        len(selected_nodes) // 2,
        len(selected_nodes) - 1
    ]

    sample_nodes = [selected_nodes[i] for i in positions]

    source = sample_nodes[0]
    targets = sample_nodes[1:]

    path_rows = []

    for target in targets:

        distance = nx.dijkstra_path_length(
            H,
            source=source,
            target=target,
            weight="length"
        )

        path = nx.dijkstra_path(
            H,
            source=source,
            target=target,
            weight="length"
        )

        path_rows.append({
            "source_node": source,
            "target_node": target,
            "distance_km": round(distance / 1000, 3),
            "path_node_count": len(path)
        })

    paths_df = pd.DataFrame(path_rows)

    print("\nDijkstra test results:")
    print(paths_df.to_string(index=False))

    paths_df.to_csv(
        OUTPUT_DIR / "dijkstra_test_results.csv",
        index=False
    )

    print("\nAnalysis completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
