import json
import networkx as nx
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

ROAD_FILE = BASE / "data/raw/road_network/colombo_road_edges.geojson"
NODE_FILE = BASE / "data/raw/road_network/colombo_road_nodes.geojson"
CANDIDATE_FILE = BASE / "data/processed/candidates/candidate_locations.csv"

with open(ROAD_FILE, encoding="utf-8") as f:
    roads = json.load(f)

with open(NODE_FILE, encoding="utf-8") as f:
    nodes = json.load(f)

candidates = pd.read_csv(CANDIDATE_FILE)

G = nx.DiGraph()

oneway_values = {}

for feature in roads["features"]:
    p = feature["properties"]

    u = int(p["u"])
    v = int(p["v"])
    length = float(p["length"])

    value = p.get("oneway")
    oneway_values[str(value)] = oneway_values.get(str(value), 0) + 1

    G.add_edge(u, v, length=length)

    if value is not True:
        G.add_edge(v, u, length=length)

print("=" * 65)
print("ROAD DIRECTION VALUE DISTRIBUTION")
print("=" * 65)
print(oneway_values)

print("\n" + "=" * 65)
print("CANDIDATE NODE INVESTIGATION")
print("=" * 65)

for candidate_id in ["C0195", "C0198"]:

    row = candidates[candidates["candidate_id"] == candidate_id]

    if row.empty:
        print("Candidate not found:", candidate_id)
        continue

    row = row.iloc[0]
    node_id = int(row["node_id"])

    print(f"\nCandidate: {candidate_id}")
    print("Node ID:", node_id)
    print("Latitude:", row["latitude"])
    print("Longitude:", row["longitude"])
    print("Source:", row["candidate_sources"])
    print("Exists in graph:", node_id in G)

    if node_id in G:
        print("Incoming directed edges:", G.in_degree(node_id))
        print("Outgoing directed edges:", G.out_degree(node_id))

        print(
            "Weak component size:",
            len(nx.node_connected_component(G.to_undirected(), node_id))
        )

        strong_component = next(
            component
            for component in nx.strongly_connected_components(G)
            if node_id in component
        )
        print("Strong component size:", len(strong_component))
# Check selected facility nodes' weak components
print("\n" + "=" * 65)
print("WEAK COMPONENT MEMBERSHIP")
print("=" * 65)

components = list(nx.weakly_connected_components(G))

component_lookup = {
    node: index
    for index, component in enumerate(components)
    for node in component
}

for candidate_id in ["C0195", "C0198"]:
    row = candidates[candidates["candidate_id"] == candidate_id].iloc[0]
    node_id = int(row["node_id"])

    print(
        candidate_id,
        "node:", node_id,
        "weak component:", component_lookup.get(node_id),
        "component size:",
        len(nx.node_connected_component(G.to_undirected(), node_id))
    )

print("\nDiagnostic completed.")

