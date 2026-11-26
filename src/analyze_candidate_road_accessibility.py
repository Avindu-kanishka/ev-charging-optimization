from pathlib import Path
import pandas as pd
import networkx as nx
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    ROOT / "data/raw/road_network/colombo_road_network.graphml"
)

CANDIDATE_PATH = (
    ROOT / "data/processed/candidates/candidate_locations.csv"
)

OUTPUT_DIR = ROOT / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading directed road network...")

    G = ox.load_graphml(filepath=GRAPH_PATH)

    print("Graph type:", type(G).__name__)
    print("Directed:", G.is_directed())
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("\nCalculating connected components...")

    weak_components = list(nx.weakly_connected_components(G))
    strong_components = list(nx.strongly_connected_components(G))

    weak_size = {}
    strong_size = {}

    for component in weak_components:
        size = len(component)
        for node in component:
            weak_size[str(node)] = size

    for component in strong_components:
        size = len(component)
        for node in component:
            strong_size[str(node)] = size

    largest_weak_size = max(map(len, weak_components))

    largest_weak_nodes = {
        str(node)
        for component in weak_components
        if len(component) == largest_weak_size
        for node in component
    }

    print("Weak components:", len(weak_components))
    print("Strong components:", len(strong_components))
    print("Largest weak component:", largest_weak_size)

    print("\nLoading candidate locations...")

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"candidate_id": str}
    )

    candidates["node_id"] = candidates["node_id"].astype(str)

    graph_nodes = {str(node) for node in G.nodes}

    records = []

    for _, row in candidates.iterrows():
        node_id = row["node_id"]
        exists = node_id in graph_nodes

        if exists:
            graph_node = next(
                node for node in G.nodes
                if str(node) == node_id
            )

            in_degree = G.in_degree(graph_node)
            out_degree = G.out_degree(graph_node)

            weak_component = weak_size[node_id]
            strong_component = strong_size[node_id]

            in_largest_weak = (
                node_id in largest_weak_nodes
            )
        else:
            in_degree = None
            out_degree = None
            weak_component = 0
            strong_component = 0
            in_largest_weak = False

        records.append({
            "candidate_id": row["candidate_id"],
            "node_id": node_id,
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "node_in_graph": exists,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": (
                in_degree + out_degree
                if exists else None
            ),
            "weak_component_size": weak_component,
            "strong_component_size": strong_component,
            "in_largest_weak_component": in_largest_weak,
            "mean_snap_distance_m": row["mean_snap_distance_m"],
            "max_snap_distance_m": row["max_snap_distance_m"],
            "snap_over_100m": row["snap_over_100m"]
        })

    result = pd.DataFrame(records)

    output_path = (
        OUTPUT_DIR / "candidate_road_accessibility_metrics.csv"
    )

    result.to_csv(output_path, index=False)

    print("\n========== CANDIDATE ACCESSIBILITY SUMMARY ==========")
    print("Total candidates:", len(result))
    print("Nodes in graph:", result["node_in_graph"].sum())
    print(
        "In largest weak component:",
        result["in_largest_weak_component"].sum()
    )
    print(
        "Zero out-degree:",
        (result["out_degree"] == 0).sum()
    )
    print(
        "Zero in-degree:",
        (result["in_degree"] == 0).sum()
    )
    print(
        "Snap over 100m:",
        result["snap_over_100m"].sum()
    )

    print("\nSmallest weak components:")
    print(
        result.sort_values("weak_component_size")[
            [
                "candidate_id",
                "node_id",
                "in_degree",
                "out_degree",
                "weak_component_size",
                "strong_component_size",
                "mean_snap_distance_m"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
