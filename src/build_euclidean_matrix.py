
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMAND_FILE = (
    PROJECT_ROOT / "data/processed/demand/road_demand_nodes.csv"
)
CANDIDATE_FILE = (
    PROJECT_ROOT / "data/processed/candidates/candidate_locations.csv"
)
ROAD_MATRIX_FILE = (
    PROJECT_ROOT / "data/processed/distance/demand_candidate_distances_km.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data/processed/distance/euclidean_distances_km.csv"
)


def haversine_matrix(demand_coords, candidate_coords):
    """Calculate pairwise straight-line distances in kilometres."""

    earth_radius_km = 6371.0088

    demand_lat = np.radians(demand_coords[:, 0])[:, None]
    demand_lon = np.radians(demand_coords[:, 1])[:, None]

    candidate_lat = np.radians(candidate_coords[:, 0])[None, :]
    candidate_lon = np.radians(candidate_coords[:, 1])[None, :]

    delta_lat = candidate_lat - demand_lat
    delta_lon = candidate_lon - demand_lon

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(demand_lat)
        * np.cos(candidate_lat)
        * np.sin(delta_lon / 2) ** 2
    )

    return (
        2
        * earth_radius_km
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def main():
    print("Loading datasets...")

    demand = pd.read_csv(DEMAND_FILE, dtype={"node_id": str})
    candidates = pd.read_csv(CANDIDATE_FILE)

    candidates["node_id"] = candidates["node_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    # Use the existing road matrix to preserve the exact row/column order.
    road_matrix = pd.read_csv(
        ROAD_MATRIX_FILE,
        index_col=0,
        dtype=str
    )

    demand_ids = road_matrix.index.astype(str).tolist()
    candidate_ids = road_matrix.columns.astype(str).tolist()

    demand_lookup = demand.set_index("node_id")

    # Determine whether the existing matrix columns use candidate_id
    # or road-network node_id.
    if set(candidate_ids).issubset(set(candidates["candidate_id"])):
        candidate_lookup = candidates.set_index("candidate_id")
    elif set(candidate_ids).issubset(set(candidates["node_id"])):
        candidate_lookup = candidates.set_index("node_id")
    else:
        raise ValueError(
            "Candidate IDs in the road matrix do not match candidate data."
        )

    missing_demand = set(demand_ids) - set(demand_lookup.index)
    missing_candidates = set(candidate_ids) - set(candidate_lookup.index)

    if missing_demand or missing_candidates:
        raise ValueError(
            f"Missing demand IDs: {len(missing_demand)}, "
            f"missing candidate IDs: {len(missing_candidates)}"
        )

    demand_aligned = demand_lookup.loc[demand_ids]
    candidate_aligned = candidate_lookup.loc[candidate_ids]

    demand_coords = demand_aligned[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)

    candidate_coords = candidate_aligned[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)

    print("Calculating Euclidean distances...")

    distances = haversine_matrix(
        demand_coords,
        candidate_coords
    )

    result = pd.DataFrame(
        distances,
        index=demand_ids,
        columns=candidate_ids
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index_label="demand_node_id")

    print("\nEUCLIDEAN DISTANCE MATRIX CREATED")
    print("Output:", OUTPUT_FILE)
    print("Matrix shape:", result.shape)
    print("Minimum distance:", round(result.min().min(), 3), "km")
    print("Maximum distance:", round(result.max().max(), 3), "km")
    print("Mean distance:", round(result.mean().mean(), 3), "km")
    print("Median distance:", round(result.stack().median(), 3), "km")


if __name__ == "__main__":
    main()
