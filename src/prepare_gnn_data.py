import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

GRAPH_PATH = "data/raw/road_network/colombo_road_network.graphml"
DATA_PATH = "data/processed/ml/spatial_demand_proxy_dataset.csv"
CV_PATH = "data/processed/ml/spatial_cv_assignments.csv"

OUTPUT_DIR = Path("data/processed/ml/gnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "colombo_gnn_dataset.npz"
REPORT_PATH = OUTPUT_DIR / "gnn_data_preparation_report.json"

print("Loading graph and datasets...")
G = nx.read_graphml(GRAPH_PATH)
df = pd.read_csv(DATA_PATH)
cv = pd.read_csv(CV_PATH)

df["node_id"] = df["node_id"].astype(str)
cv["node_id"] = cv["node_id"].astype(str)

if df["node_id"].duplicated().any():
    raise ValueError("Duplicate node IDs in ML dataset.")

if cv["node_id"].duplicated().any():
    raise ValueError("Duplicate node IDs in CV assignments.")

node_ids = sorted(map(str, G.nodes()))
graph_nodes = set(node_ids)

if set(df["node_id"]) != graph_nodes:
    raise ValueError("Graph and ML dataset node IDs do not match.")

if set(cv["node_id"]) != graph_nodes:
    raise ValueError("Graph and CV node IDs do not match.")

df = df.set_index("node_id").loc[node_ids].reset_index()
cv = cv.set_index("node_id").loc[node_ids].reset_index()

feature_names = [
    c for c in df.columns
    if c not in ["node_id", "demand_proxy_count"]
]

X = df[feature_names].apply(pd.to_numeric, errors="raise").to_numpy(
    dtype=np.float32
)
y = df["demand_proxy_count"].to_numpy(dtype=np.float32)

if not np.isfinite(X).all():
    raise ValueError("Feature matrix contains missing or infinite values.")

if not np.isfinite(y).all():
    raise ValueError("Target contains missing or infinite values.")

if not np.array_equal(
    y,
    cv["demand_proxy_count"].to_numpy(dtype=np.float32)
):
    raise ValueError("Target mismatch between ML dataset and CV assignments.")

node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

print("Building undirected road adjacency...")
undirected_edges = set()

for u, v in G.edges():
    u = str(u)
    v = str(v)

    if u == v:
        continue

    i = node_to_index[u]
    j = node_to_index[v]

    undirected_edges.add((min(i, j), max(i, j)))

directed_pairs = set()

for i, j in undirected_edges:
    directed_pairs.add((i, j))
    directed_pairs.add((j, i))

edge_index = np.asarray(
    sorted(directed_pairs),
    dtype=np.int64
).T

cv_fold = cv["cv_fold"].to_numpy(dtype=np.int64)
spatial_block = cv["spatial_block"].astype(str).to_numpy()

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y,
    edge_index=edge_index,
    node_ids=np.asarray(node_ids, dtype=str),
    feature_names=np.asarray(feature_names, dtype=str),
    cv_fold=cv_fold,
    spatial_block=spatial_block
)

# Reload saved file to validate the actual artifact.
saved = np.load(OUTPUT_PATH, allow_pickle=False)

X_check = saved["X"]
y_check = saved["y"]
edges_check = saved["edge_index"]

assert X_check.shape == (5453, len(feature_names))
assert y_check.shape == (5453,)
assert edges_check.shape[0] == 2
assert len(saved["node_ids"]) == 5453
assert len(saved["cv_fold"]) == 5453
assert np.isfinite(X_check).all()
assert np.isfinite(y_check).all()

edge_pairs = set(map(tuple, edges_check.T.tolist()))

for i, j in edge_pairs:
    assert (j, i) in edge_pairs
    assert i != j

report = {
    "nodes": int(len(node_ids)),
    "features": int(len(feature_names)),
    "feature_names": feature_names,
    "target_name": "demand_proxy_count",
    "target_positive_nodes": int((y > 0).sum()),
    "target_zero_nodes": int((y == 0).sum()),
    "target_total": float(y.sum()),
    "undirected_road_edges": int(len(undirected_edges)),
    "bidirectional_adjacency_edges": int(edge_index.shape[1]),
    "spatial_fold_distribution": {
        str(k): int(v)
        for k, v in pd.Series(cv_fold).value_counts().sort_index().items()
    },
    "feature_missing_values": int(np.isnan(X).sum()),
    "feature_infinite_values": int(np.isinf(X).sum()),
    "validation": "passed"
}

REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== GNN DATA PREPARATION REPORT ===")
print(json.dumps(report, indent=2))
print("\nSaved:", OUTPUT_PATH)
print("Saved:", REPORT_PATH)
print("\nGNN DATA PREPARATION VALIDATED")
