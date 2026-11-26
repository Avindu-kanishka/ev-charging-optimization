from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data/raw/road_network/colombo_road_network.graphml"
DEMAND_PATH = ROOT / "data/processed/demand/road_demand_nodes.csv"
CANDIDATE_PATH = ROOT / "data/processed/candidates/candidate_locations.csv"
DISTANCE_PATH = ROOT / "data/processed/distance/demand_candidate_distances_km.csv"
REPORT_PATH = ROOT / "outputs/analysis/evision_data_quality_report.json"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required project input is missing: {path}")


def main() -> None:
    print("=" * 68)
    print("EVISION AI - BASELINE DATA QUALITY AUDIT")
    print("=" * 68)

    for path in (GRAPH_PATH, DEMAND_PATH, CANDIDATE_PATH, DISTANCE_PATH):
        require_file(path)

    graph = ox.load_graphml(GRAPH_PATH)
    demand = pd.read_csv(DEMAND_PATH, dtype={"node_id": str})
    candidates = pd.read_csv(CANDIDATE_PATH, dtype={"node_id": str, "candidate_id": str})
    distances = pd.read_csv(DISTANCE_PATH, index_col=0)

    distances.index = distances.index.astype(str)
    distances.columns = distances.columns.astype(str)

    required_demand = {"node_id", "demand_weight", "longitude", "latitude"}
    required_candidates = {"node_id", "candidate_id", "longitude", "latitude"}
    missing_demand_columns = sorted(required_demand - set(demand.columns))
    missing_candidate_columns = sorted(required_candidates - set(candidates.columns))
    if missing_demand_columns:
        raise ValueError(f"Demand file missing columns: {missing_demand_columns}")
    if missing_candidate_columns:
        raise ValueError(f"Candidate file missing columns: {missing_candidate_columns}")

    demand["node_id"] = demand["node_id"].astype(str)
    candidates["node_id"] = candidates["node_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    graph_node_ids = {str(node_id) for node_id in graph.nodes}
    demand_ids = set(demand["node_id"])
    candidate_ids = set(candidates["candidate_id"])
    matrix_demand_ids = set(distances.index)
    matrix_candidate_ids = set(distances.columns)

    matrix = distances.to_numpy(dtype=float)
    finite = np.isfinite(matrix)

    demand_overlap = demand_ids.intersection(set(candidates["node_id"]))
    demand_missing_from_graph = sorted(demand_ids - graph_node_ids)
    candidates_missing_from_graph = sorted(set(candidates["node_id"]) - graph_node_ids)

    report = {
        "graph": {
            "type": type(graph).__name__,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "weakly_connected_components": (
                nx.number_weakly_connected_components(graph)
                if graph.is_directed()
                else nx.number_connected_components(graph)
            ),
        },
        "demand": {
            "rows": int(len(demand)),
            "unique_node_ids": int(demand["node_id"].nunique()),
            "total_proxy_weight": float(pd.to_numeric(demand["demand_weight"], errors="coerce").sum()),
            "duplicate_node_ids": int(demand["node_id"].duplicated().sum()),
            "nodes_missing_from_graph": len(demand_missing_from_graph),
            "nodes_missing_from_graph_sample": demand_missing_from_graph[:10],
            "snap_over_100m": (
                int((pd.to_numeric(demand["max_snap_distance_m"], errors="coerce") > 100).sum())
                if "max_snap_distance_m" in demand.columns else None
            ),
        },
        "candidates": {
            "rows": int(len(candidates)),
            "unique_candidate_ids": int(candidates["candidate_id"].nunique()),
            "unique_road_nodes": int(candidates["node_id"].nunique()),
            "duplicate_candidate_ids": int(candidates["candidate_id"].duplicated().sum()),
            "road_nodes_missing_from_graph": len(candidates_missing_from_graph),
            "road_nodes_missing_from_graph_sample": candidates_missing_from_graph[:10],
            "snap_over_100m": (
                int((pd.to_numeric(candidates["max_snap_distance_m"], errors="coerce") > 100).sum())
                if "max_snap_distance_m" in candidates.columns else None
            ),
        },
        "distance_matrix": {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "row_labels_match_demand_set": matrix_demand_ids == demand_ids,
            "column_labels_match_candidate_id_set": matrix_candidate_ids == candidate_ids,
            "rows_missing_from_demand": len(matrix_demand_ids - demand_ids),
            "demand_rows_missing_from_matrix": len(demand_ids - matrix_demand_ids),
            "columns_missing_from_candidates": len(matrix_candidate_ids - candidate_ids),
            "candidate_ids_missing_from_matrix": len(candidate_ids - matrix_candidate_ids),
            "reachable_pairs": int(finite.sum()),
            "unreachable_pairs": int((~finite).sum()),
            "negative_finite_distances": int((matrix[finite] < 0).sum()),
            "mean_reachable_distance_km": float(matrix[finite].mean()) if finite.any() else None,
            "median_reachable_distance_km": float(np.median(matrix[finite])) if finite.any() else None,
        },
        "candidate_demand_overlap": {
            "shared_road_nodes": len(demand_overlap),
            "shared_road_node_sample": sorted(demand_overlap)[:10],
            "warning": (
                "Candidate and demand nodes overlap. Because candidate sites are selected "
                "from OSM activity features that may also contribute to demand proxies, "
                "coverage results may be optimistic and require a sensitivity analysis."
            ) if demand_overlap else None,
        },
    }

    report["checks"] = {
        "demand_node_ids_unique": report["demand"]["duplicate_node_ids"] == 0,
        "candidate_ids_unique": report["candidates"]["duplicate_candidate_ids"] == 0,
        "demand_nodes_exist_in_graph": len(demand_missing_from_graph) == 0,
        "candidate_nodes_exist_in_graph": len(candidates_missing_from_graph) == 0,
        "distance_matrix_labels_aligned": (
            report["distance_matrix"]["row_labels_match_demand_set"]
            and report["distance_matrix"]["column_labels_match_candidate_id_set"]
        ),
        "no_negative_distances": report["distance_matrix"]["negative_finite_distances"] == 0,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Graph: {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges")
    print(f"Demand nodes: {len(demand):,}; proxy weight: {report['demand']['total_proxy_weight']:,.0f}")
    print(f"Candidate locations: {len(candidates):,}")
    print(f"Distance matrix: {matrix.shape}; reachable pairs: {finite.sum():,}; unreachable: {(~finite).sum():,}")
    print(f"Candidate-demand shared road nodes: {len(demand_overlap):,}")
    print("\nCHECKS")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'REVIEW'}  {name}")
    print(f"\nAudit report saved to: {REPORT_PATH}")
    print("Note: this audit reads existing data and writes only the JSON report.")


if __name__ == "__main__":
    main()
