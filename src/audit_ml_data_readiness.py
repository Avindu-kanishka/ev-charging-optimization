from pathlib import Path
import pandas as pd
import networkx as nx
import json

root = Path(__file__).resolve().parents[1]

original = pd.read_csv(root / "data/processed/demand/road_demand_nodes.csv")
filtered = pd.read_csv(root / "data/processed/demand/non_candidate_demand_nodes.csv")
candidates = pd.read_csv(root / "data/processed/candidates/candidate_locations.csv")

graph = nx.read_graphml(root / "data/raw/road_network/colombo_road_network.graphml")

weights = filtered["demand_weight"]

report = {
    "project": "EV Charging Infrastructure Planning - Layer 2",
    "audit": "ML dataset readiness",
    "recommended_initial_target": "demand_weight",
    "recommended_initial_dataset": "non_candidate_demand_nodes.csv",
    "target_interpretation": "OSM-based spatial demand proxy, not actual EV charging demand",
    "original_dataset": {
        "nodes": len(original),
        "total_proxy_weight": int(original["demand_weight"].sum()),
        "missing_values": int(original.isna().sum().sum()),
        "duplicate_node_ids": int(original["node_id"].duplicated().sum()),
        "demand_weight_equals_proxy_count": bool(
            original["demand_weight"].equals(original["proxy_count"])
        )
    },
    "filtered_dataset": {
        "nodes": len(filtered),
        "total_proxy_weight": int(weights.sum()),
        "missing_values": int(filtered.isna().sum().sum()),
        "duplicate_node_ids": int(filtered["node_id"].duplicated().sum()),
        "minimum_weight": int(weights.min()),
        "maximum_weight": int(weights.max()),
        "mean_weight": float(weights.mean()),
        "median_weight": float(weights.median()),
        "standard_deviation": float(weights.std()),
        "weight_distribution": {
            str(k): int(v)
            for k, v in weights.value_counts().sort_index().items()
        }
    },
    "candidate_dataset": {
        "candidate_locations": len(candidates),
        "duplicate_node_ids": int(candidates["node_id"].duplicated().sum()),
        "overlap_with_original_demand_nodes": int(
            original["node_id"].isin(candidates["node_id"]).sum()
        )
    },
    "road_graph": {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed()
    },
    "available_features": [
        "longitude",
        "latitude",
        "mean_snap_distance_m",
        "max_snap_distance_m",
        "road node degree",
        "street count",
        "road network centrality",
        "local road density"
    ],
    "feature_leakage_warning": (
        "POI category counts contributing to the demand target must not "
        "be treated as independent predictors without a leakage assessment."
    ),
    "actual_ev_data": {
        "identified": False,
        "required_future_data": [
            "charging session records",
            "energy consumption",
            "station utilization",
            "timestamped charging demand"
        ]
    }
}

output = root / "outputs/analysis/ml_data_readiness_report.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=4), encoding="utf-8")

print("ML DATA READINESS AUDIT COMPLETED")
print("Recommended dataset:", report["recommended_initial_dataset"])
print("Filtered nodes:", report["filtered_dataset"]["nodes"])
print("Filtered proxy weight:", report["filtered_dataset"]["total_proxy_weight"])
print("Road graph nodes:", report["road_graph"]["nodes"])
print("Actual EV data identified:", report["actual_ev_data"]["identified"])
print("Report:", output)
