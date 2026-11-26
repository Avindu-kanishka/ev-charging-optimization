import json
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

ROAD_FILE = BASE / "data/raw/road_network/colombo_road_edges.geojson"
NODE_FILE = BASE / "data/raw/road_network/colombo_road_nodes.geojson"
SELECTED_FILE = BASE / "outputs/optimization/mclp_facility_separation_selected_locations.csv"

with open(ROAD_FILE, encoding="utf-8") as f:
    roads = json.load(f)

with open(NODE_FILE, encoding="utf-8") as f:
    road_nodes = json.load(f)

selected = pd.read_csv(SELECTED_FILE)

G = nx.MultiDiGraph()

for feature in roads["features"]:
    props = feature["properties"]

    u = int(props["u"])
    v = int(props["v"])
    length = float(props["length"])

    G.add_edge(u, v, length=length)

    # Add reverse edge for roads that are not one-way
    if props["oneway"] is False:
        G.add_edge(v, u, length=length)

selected_nodes = set(selected["node_id"].astype(int))
graph_nodes = set(G.nodes())

print("=" * 60)
print("ROAD NETWORK VALIDATION")
print("=" * 60)

print("GeoJSON road edges:", len(roads["features"]))
print("GeoJSON road nodes:", len(road_nodes["features"]))
print("Graph nodes:", G.number_of_nodes())
print("Directed graph edges:", G.number_of_edges())

print("\nSelected unique facility nodes:", len(selected_nodes))
print("Selected nodes in graph:", len(selected_nodes & graph_nodes))
print("Selected nodes missing from graph:", len(selected_nodes - graph_nodes))

print("\nStrongly connected components:", nx.number_strongly_connected_components(G))
print("Weakly connected components:", nx.number_weakly_connected_components(G))

oneway_count = sum(
    1 for f in roads["features"]
    if f["properties"]["oneway"] is True
)

print("\nOne-way road features:", oneway_count)
print("Non-one-way road features:", len(roads["features"]) - oneway_count)

print("\nValidation completed.")
