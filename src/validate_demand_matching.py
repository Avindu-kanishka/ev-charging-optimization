
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_road_network.graphml"
)

DEMAND_PATH = (
    BASE_DIR / "data" / "raw" / "demand"
    / "colombo_demand_proxies.geojson"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading graph and demand locations...")

    G = ox.load_graphml(GRAPH_PATH)
    demand = gpd.read_file(DEMAND_PATH).to_crs("EPSG:4326")

    # Repeat matching to retain individual feature details
    X = demand.geometry.x.to_numpy()
    Y = demand.geometry.y.to_numpy()

    nearest_nodes, distances = ox.distance.nearest_nodes(
        G, X=X, Y=Y, return_dist=True
    )

    demand["nearest_node"] = nearest_nodes
    demand["snap_distance_m"] = distances

    # Distance distribution
    print("\nSNAPPING DISTANCE DISTRIBUTION")

    print(
        demand["snap_distance_m"]
        .describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99])
        .round(2)
    )

    # Inspect features over 100m
    outliers = demand[
        demand["snap_distance_m"] > 100
    ].copy()

    columns = [
        c for c in [
            "name", "amenity", "shop", "office",
            "nearest_node", "snap_distance_m"
        ]
        if c in outliers.columns
    ]

    outliers[columns].to_csv(
        OUTPUT_DIR / "demand_snapping_outliers.csv",
        index=False
    )

    # Category-level distance summary
    category_rows = []

    for category in ["amenity", "shop", "office"]:

        if category not in demand.columns:
            continue

        summary = (
            demand.groupby(category, dropna=True)
            .agg(
                feature_count=("snap_distance_m", "size"),
                mean_distance_m=("snap_distance_m", "mean"),
                max_distance_m=("snap_distance_m", "max"),
                over_100m=("snap_distance_m",
                           lambda x: (x > 100).sum())
            )
            .reset_index()
            .rename(columns={category: "category"})
        )

        summary["feature_type"] = category
        category_rows.append(summary)

    if category_rows:
        category_summary = pd.concat(
            category_rows, ignore_index=True
        )

        category_summary.to_csv(
            OUTPUT_DIR / "demand_category_quality.csv",
            index=False
        )

    print("\nFeatures over 100m:", len(outliers))
    print("Saved outliers to:", OUTPUT_DIR)
    print("Validation completed.")


if __name__ == "__main__":
    main()
