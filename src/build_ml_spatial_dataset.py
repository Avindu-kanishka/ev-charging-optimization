from pathlib import Path
import json
import math
import re

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = ROOT / "data/raw/road_network/colombo_road_network.graphml"
DEMAND_PATH = ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"

OUTPUT_DIR = ROOT / "data/processed/ml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "spatial_demand_proxy_dataset.csv"
REPORT_PATH = ROOT / "outputs/analysis/ml_spatial_dataset_report.json"

print("Loading road graph...")
G = nx.read_graphml(GRAPH_PATH)

print("Loading filtered demand observations...")
demand = pd.read_csv(DEMAND_PATH)

# Convert OSM graph node IDs to integers.
records = []

for node_id, attrs in G.nodes(data=True):
    try:
        node_id_int = int(node_id)
        lon = float(attrs["x"])
        lat = float(attrs["y"])
    except (ValueError, KeyError, TypeError):
        continue

    records.append({
        "node_id": node_id_int,
        "longitude": lon,
        "latitude": lat,
        "street_count": float(attrs.get("street_count", 0) or 0)
    })

nodes = pd.DataFrame(records)

if nodes["node_id"].duplicated().any():
    raise ValueError("Duplicate graph node IDs detected.")

demand_map = demand.set_index("node_id")["demand_weight"]

unmatched = set(demand_map.index) - set(nodes["node_id"])

if unmatched:
    raise ValueError(
        f"{len(unmatched)} demand nodes are missing from the road graph."
    )

nodes["demand_proxy_count"] = (
    nodes["node_id"].map(demand_map).fillna(0).astype(int)
)

# Aggregate independent road-network features.
print("Engineering road-network features...")

road_types = [
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "residential", "service", "unclassified",
    "living_street"
]

features = {
    int(row.node_id): {
        "in_degree": 0,
        "out_degree": 0,
        "incident_edge_count": 0,
        "incident_road_length_sum_m": 0.0,
        "incident_road_length_mean_m": 0.0,
        "incident_road_length_max_m": 0.0,
        **{f"road_{r}_count": 0 for r in road_types},
        "road_other_count": 0
    }
    for row in nodes.itertuples()
}

incident_lengths = {node_id: [] for node_id in features}

for u, v, attrs in G.edges(data=True):
    try:
        u = int(u)
        v = int(v)
    except (ValueError, TypeError):
        continue

    if u not in features or v not in features:
        continue

    try:
        length = float(attrs.get("length", 0) or 0)
    except (ValueError, TypeError):
        length = 0.0

    features[u]["out_degree"] += 1
    features[v]["in_degree"] += 1

    for node_id in (u, v):
        features[node_id]["incident_edge_count"] += 1
        features[node_id]["incident_road_length_sum_m"] += length
        incident_lengths[node_id].append(length)

    highway = attrs.get("highway", "other")

    if isinstance(highway, list):
        highway_values = highway
    else:
        highway_values = [highway]

    for highway_value in highway_values:
        highway_value = str(highway_value).lower()

        if highway_value in road_types:
            features[u][f"road_{highway_value}_count"] += 1
            features[v][f"road_{highway_value}_count"] += 1
        else:
            features[u]["road_other_count"] += 1
            features[v]["road_other_count"] += 1

for node_id, values in features.items():
    lengths = incident_lengths[node_id]

    if lengths:
        values["incident_road_length_mean_m"] = float(np.mean(lengths))
        values["incident_road_length_max_m"] = float(np.max(lengths))

    values["degree"] = (
        values["in_degree"] + values["out_degree"]
    )

road_features = pd.DataFrame.from_dict(features, orient="index")
road_features.index.name = "node_id"
road_features = road_features.reset_index()

nodes = nodes.merge(
    road_features,
    on="node_id",
    how="left",
    validate="one_to_one"
)

# Calculate local graph-node density using geographic distance.
print("Calculating local spatial density...")

earth_radius_m = 6371000.0

coordinates_rad = np.radians(
    nodes[["latitude", "longitude"]].to_numpy()
)

tree = BallTree(coordinates_rad, metric="haversine")

for radius_m in (250, 500):
    radius_rad = radius_m / earth_radius_m

    counts = tree.query_radius(
        coordinates_rad,
        r=radius_rad,
        count_only=True
    )

    # Exclude the node itself.
    nodes[f"local_node_count_{radius_m}m"] = counts - 1

# Remove constant predictors that cannot help distinguish demand.
identity_target_columns = {"node_id", "demand_proxy_count"}

constant_columns = [
    column for column in nodes.columns
    if column not in identity_target_columns
    and nodes[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("Removing constant predictors:", constant_columns)
    nodes = nodes.drop(columns=constant_columns)

# Final checks.
feature_columns = [
    column for column in nodes.columns
    if column not in ("node_id", "demand_proxy_count")
]

forbidden = {
    "demand_weight",
    "proxy_count",
    "mean_snap_distance_m",
    "max_snap_distance_m"
}

if forbidden.intersection(feature_columns):
    raise ValueError("Target leakage detected in predictor columns.")

if nodes.isna().any().any():
    raise ValueError("Missing values detected in final dataset.")

if (nodes["demand_proxy_count"] < 0).any():
    raise ValueError("Negative target values detected.")

if int(nodes["demand_proxy_count"].sum()) != int(demand["demand_weight"].sum()):
    raise ValueError("Target total does not match the filtered demand dataset.")

nodes = nodes.sort_values("node_id").reset_index(drop=True)

nodes.to_csv(OUTPUT_PATH, index=False)

positive_nodes = int((nodes["demand_proxy_count"] > 0).sum())
zero_nodes = int((nodes["demand_proxy_count"] == 0).sum())

report = {
    "dataset": str(OUTPUT_PATH.relative_to(ROOT)),
    "graph_nodes": G.number_of_nodes(),
    "graph_edges": G.number_of_edges(),
    "dataset_rows": len(nodes),
    "positive_demand_nodes": positive_nodes,
    "zero_demand_nodes": zero_nodes,
    "total_demand_proxy_count": int(nodes["demand_proxy_count"].sum()),
    "target_column": "demand_proxy_count",
    "predictor_count": len(feature_columns),
    "constant_features_removed": constant_columns,
    "predictor_columns": feature_columns,
    "missing_values": int(nodes.isna().sum().sum()),
    "duplicate_node_ids": int(nodes["node_id"].duplicated().sum()),
    "target_leakage_check": "PASSED",
    "actual_ev_demand": False
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

print("\nSPATIAL ML DATASET CREATED")
print("Rows:", len(nodes))
print("Positive demand nodes:", positive_nodes)
print("Zero demand nodes:", zero_nodes)
print("Total proxy count:", int(nodes["demand_proxy_count"].sum()))
print("Predictor count:", len(feature_columns))
print("Missing values:", int(nodes.isna().sum().sum()))
print("Target leakage check: PASSED")
print("Dataset:", OUTPUT_PATH)
print("Report:", REPORT_PATH)
