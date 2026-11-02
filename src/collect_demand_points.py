
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_study_boundary.geojson"
)

OUTPUT_DIR = BASE_DIR / "data" / "raw" / "demand"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OSM features representing urban activity
TAGS = {
    "amenity": [
        "hospital",
        "clinic",
        "school",
        "university",
        "college",
        "marketplace",
        "fuel",
        "parking"
    ],
    "shop": True,
    "office": True
}


def main():

    print("=" * 60)
    print("COLLECTING URBAN ACTIVITY DEMAND PROXIES")
    print("=" * 60)

    # Load study boundary
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
    polygon = boundary.geometry.union_all()

    print("\nDownloading OSM activity locations...")

    features = ox.features_from_polygon(
        polygon,
        tags=TAGS
    )

    print("Downloaded features:", len(features))

    # Convert to geographic coordinates
    features = features.reset_index()

    # Keep valid geometries
    features = features[
        features.geometry.notna()
        & ~features.geometry.is_empty
    ].copy()

    # Use representative points for polygons
    features["geometry"] = features.geometry.representative_point()

    # Extract useful attributes
    available_columns = [
        col for col in [
            "name",
            "amenity",
            "shop",
            "office",
            "geometry"
        ]
        if col in features.columns
    ]

    demand = features[available_columns].copy()

    demand = gpd.GeoDataFrame(
        demand,
        geometry="geometry",
        crs="EPSG:4326"
    )

    demand = demand.reset_index(drop=True)
    demand["demand_id"] = range(1, len(demand) + 1)

    # Classify proxy categories
    def classify(row):

        for column in ["amenity", "shop", "office"]:
            value = row.get(column)

            if pd.notna(value):
                return f"{column}_{value}"

        return "other"

    demand["proxy_category"] = demand.apply(
        classify,
        axis=1
    )

    # Save data
    output_path = OUTPUT_DIR / "colombo_demand_proxies.geojson"

    demand.to_file(output_path, driver="GeoJSON")

    summary = (
        demand.groupby("proxy_category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    summary.to_csv(
        OUTPUT_DIR / "demand_proxy_summary.csv",
        index=False
    )

    print("\nDemand proxy summary:")
    print(summary.to_string(index=False))

    print("\nTotal demand proxy locations:", len(demand))
    print("Saved:", output_path)

    print("\nCollection completed.")


if __name__ == "__main__":
    main()
