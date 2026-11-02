
from pathlib import Path

import osmnx as ox
import networkx as nx

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "road_network"
OUTPUT_DIR = BASE_DIR / "outputs" / "maps"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 180

# IMPORTANT:
# This is a provisional Colombo boundary, not yet verified
# against the official Colombo Municipal Council boundary.
PLACE = "Colombo, Sri Lanka"


def main():

    print("=" * 60)
    print("COLOMBO ROAD NETWORK COLLECTION")
    print("=" * 60)

    # 1. Download study boundary
    print("\n[1/4] Downloading Colombo study boundary...")

    boundary = ox.geocode_to_gdf(PLACE)

    if boundary.empty:
        raise RuntimeError("Could not retrieve Colombo boundary.")

    polygon = boundary.geometry.union_all()

    if polygon.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Invalid boundary geometry: {polygon.geom_type}")

    print("Boundary retrieved successfully.")
    print("Geometry:", polygon.geom_type)
    print("Bounds:", polygon.bounds)

    # 2. Download driving road network
    print("\n[2/4] Downloading road network from OpenStreetMap...")
    print("This may take several minutes.")

    graph = ox.graph_from_polygon(
        polygon,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True
    )

    if graph.number_of_nodes() == 0:
        raise RuntimeError("Downloaded road network is empty.")

    # 3. Analyse graph
    print("\n[3/4] Analysing road network...")

    nodes_count = graph.number_of_nodes()
    edges_count = graph.number_of_edges()

    components = list(nx.weakly_connected_components(graph))

    total_length = sum(
        data["length"]
        for _, _, _, data in graph.edges(keys=True, data=True)
    )

    print(f"Nodes: {nodes_count:,}")
    print(f"Edges: {edges_count:,}")
    print(f"Weakly connected components: {len(components)}")
    print(f"Total road length: {total_length / 1000:.2f} km")

    # 4. Save files
    print("\n[4/4] Saving road network files...")

    graph_path = RAW_DIR / "colombo_road_network.graphml"
    nodes_path = RAW_DIR / "colombo_road_nodes.geojson"
    edges_path = RAW_DIR / "colombo_road_edges.geojson"
    boundary_path = RAW_DIR / "colombo_study_boundary.geojson"

    ox.save_graphml(graph, graph_path)

    nodes, edges = ox.graph_to_gdfs(graph)

    nodes.to_file(nodes_path, driver="GeoJSON")
    edges.to_file(edges_path, driver="GeoJSON")
    boundary.to_file(boundary_path, driver="GeoJSON")

    print("\nSUCCESS! Road network collection completed.")

    print("\nGenerated files:")
    print(graph_path)
    print(nodes_path)
    print(edges_path)
    print(boundary_path)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
