
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_road_network.graphml"
)

FEATURE_PATH = (
    BASE_DIR / "data" / "raw" / "demand"
    / "colombo_demand_proxies.geojson"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("CANDIDATE CHARGING LOCATION COLLECTION")
    print("=" * 60)

    G = ox.load_graphml(GRAPH_PATH)
    features = gpd.read_file(FEATURE_PATH).to_crs("EPSG:4326")

    # Preliminary candidate-site categories
    selected = (
        features["amenity"].isin(
            ["parking", "fuel", "marketplace"]
        )
        | features["shop"].isin(
            ["mall", "supermarket", "shopping_mall"]
        )
    )

    candidates = features[selected].copy()

    print("\nCandidate source features:", len(candidates))

    # Match candidate sites to nearest road nodes
    X = candidates.geometry.x.to_numpy()
    Y = candidates.geometry.y.to_numpy()

    nearest, distances = ox.distance.nearest_nodes(
        G, X=X, Y=Y, return_dist=True
    )

    candidates["node_id"] = nearest
    candidates["snap_distance_m"] = distances

    # Identify source type
    def get_source(row):
        for col in ["amenity", "shop"]:
            value = row.get(col)
            if pd.notna(value):
                return f"{col}_{value}"
        return "other"

    candidates["candidate_source"] = candidates.apply(
        get_source, axis=1
    )

    # Keep all matched candidate nodes, aggregating duplicate nodes
    aggregated = (
        candidates.groupby("node_id")
        .agg(
            source_feature_count=("node_id", "size"),
            mean_snap_distance_m=("snap_distance_m", "mean"),
            max_snap_distance_m=("snap_distance_m", "max"),
            candidate_sources=("candidate_source",
                               lambda x: ";".join(sorted(set(x))))
        )
        .reset_index()
    )

    # Add road-node coordinates
    coordinates = pd.DataFrame([
        {
            "node_id": node,
            "longitude": G.nodes[node]["x"],
            "latitude": G.nodes[node]["y"]
        }
        for node in aggregated["node_id"]
    ])

    aggregated = aggregated.merge(
        coordinates, on="node_id", how="left"
    )

    aggregated["candidate_id"] = [
        f"C{i:04d}" for i in range(1, len(aggregated) + 1)
    ]

    aggregated["snap_over_100m"] = (
        aggregated["max_snap_distance_m"] > 100
    )

    # Save candidate locations
    csv_path = OUTPUT_DIR / "candidate_locations.csv"
    geojson_path = OUTPUT_DIR / "candidate_locations.geojson"

    aggregated.to_csv(csv_path, index=False)

    candidate_gdf = gpd.GeoDataFrame(
        aggregated,
        geometry=gpd.points_from_xy(
            aggregated.longitude,
            aggregated.latitude
        ),
        crs="EPSG:4326"
    )

    candidate_gdf.to_file(geojson_path, driver="GeoJSON")

    print("\nCANDIDATE SUMMARY")
    print("-" * 40)
    print("Source features:", len(candidates))
    print("Unique candidate road nodes:", len(aggregated))
    print(
        "Candidates with snap distance >100m:",
        aggregated["snap_over_100m"].sum()
    )

    print("\nCandidate source summary:")
    print(
        candidates["candidate_source"]
        .value_counts()
        .to_string()
    )

    print("\nSaved:", csv_path)
    print("Saved:", geojson_path)
    print("\nCandidate collection completed.")


if __name__ == "__main__":
    main()
