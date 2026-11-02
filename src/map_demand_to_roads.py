
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

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "demand"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("DEMAND TO ROAD NETWORK MATCHING")
    print("=" * 60)

    # Load graph and demand data
    print("\n[1/4] Loading data...")

    G = ox.load_graphml(GRAPH_PATH)
    demand = gpd.read_file(DEMAND_PATH).to_crs("EPSG:4326")

    print("Road nodes:", G.number_of_nodes())
    print("Demand features:", len(demand))

    # Match each demand feature to its nearest road node
    print("\n[2/4] Matching demand locations to road nodes...")

    X = demand.geometry.x.to_numpy()
    Y = demand.geometry.y.to_numpy()

    nearest_nodes, distances = ox.distance.nearest_nodes(
        G,
        X=X,
        Y=Y,
        return_dist=True
    )

    demand["nearest_node"] = nearest_nodes
    demand["snap_distance_m"] = distances

    print("Matching completed.")

    # Aggregate demand proxies at road nodes
    print("\n[3/4] Aggregating demand by road node...")

    demand["proxy_count"] = 1

    aggregated = (
        demand.groupby("nearest_node")
        .agg(
            demand_weight=("proxy_count", "sum"),
            proxy_count=("demand_id", "count"),
            mean_snap_distance_m=("snap_distance_m", "mean"),
            max_snap_distance_m=("snap_distance_m", "max")
        )
        .reset_index()
        .rename(columns={"nearest_node": "node_id"})
    )

    # Add road node coordinates
    node_rows = []

    for node_id in aggregated["node_id"]:
        node_data = G.nodes[node_id]

        node_rows.append({
            "node_id": node_id,
            "longitude": node_data["x"],
            "latitude": node_data["y"]
        })

    node_coordinates = pd.DataFrame(node_rows)

    aggregated = aggregated.merge(
        node_coordinates,
        on="node_id",
        how="left"
    )

    # Save outputs
    print("\n[4/4] Saving results...")

    output_csv = OUTPUT_DIR / "road_demand_nodes.csv"
    output_geojson = OUTPUT_DIR / "road_demand_nodes.geojson"

    aggregated.to_csv(output_csv, index=False)

    demand_points = gpd.GeoDataFrame(
        aggregated,
        geometry=gpd.points_from_xy(
            aggregated["longitude"],
            aggregated["latitude"]
        ),
        crs="EPSG:4326"
    )

    demand_points.to_file(output_geojson, driver="GeoJSON")

    # Quality checks
    print("\nDEMAND MATCHING SUMMARY")
    print("-" * 40)

    print("Original demand features:", len(demand))
    print("Unique demand road nodes:", len(aggregated))
    print(
        "Demand features matched:",
        demand["nearest_node"].notna().sum()
    )

    print(
        "Average snapping distance:",
        round(demand["snap_distance_m"].mean(), 2), "m"
    )

    print(
        "Maximum snapping distance:",
        round(demand["snap_distance_m"].max(), 2), "m"
    )

    print(
        "Features snapped over 100m:",
        (demand["snap_distance_m"] > 100).sum()
    )

    print("\nSaved:", output_csv)
    print("Saved:", output_geojson)

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
