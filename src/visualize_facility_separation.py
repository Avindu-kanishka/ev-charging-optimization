from pathlib import Path
import json

import pandas as pd
import folium

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = BASE / "data/raw/road_network/colombo_study_boundary.geojson"
ROAD_FILE = BASE / "data/raw/road_network/colombo_road_edges.geojson"
SELECTED_FILE = BASE / "outputs/optimization/mclp_facility_separation_selected_locations.csv"
DEMAND_FILE = BASE / "data/processed/demand/non_candidate_demand_nodes.csv"

OUTPUT_DIR = BASE / "outputs/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    0.00: "red",
    0.25: "blue",
    0.50: "green",
    1.00: "purple",
    1.50: "orange",
}

# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

print("Loading Colombo geographical datasets...")

with open(BOUNDARY_FILE, encoding="utf-8") as f:
    boundary = json.load(f)

with open(ROAD_FILE, encoding="utf-8") as f:
    roads = json.load(f)

selected = pd.read_csv(SELECTED_FILE)
demand = pd.read_csv(DEMAND_FILE)

demand = demand.dropna(subset=["longitude", "latitude"])
selected = selected.dropna(
    subset=["longitude", "latitude", "minimum_separation_km"]
)

print("Boundary features:", len(boundary["features"]))
print("Road edges:", len(roads["features"]))
print("Demand proxy nodes:", len(demand))
print("Selected facility records:", len(selected))

# Check study boundary bounds
properties = boundary["features"][0]["properties"]

west = float(properties["bbox_west"])
south = float(properties["bbox_south"])
east = float(properties["bbox_east"])
north = float(properties["bbox_north"])

# --------------------------------------------------
# 3. Generate maps for P=5 and P=10
# --------------------------------------------------

for p_value in [5, 10]:

    print(f"\nGenerating spatial map for P={p_value}...")

    scenario_data = selected[selected["p"] == p_value]

    if scenario_data.empty:
        raise ValueError(f"No selected facilities found for P={p_value}")

    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    # No external tiles: fully offline-compatible base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None,
        control_scale=True
    )

    # --------------------------------------------------
    # Colombo study boundary
    # --------------------------------------------------

    boundary_layer = folium.FeatureGroup(
        name="Colombo Study Area Boundary",
        show=True
    )

    folium.GeoJson(
        boundary,
        name="Study Boundary",
        style_function=lambda feature: {
            "color": "#17365D",
            "weight": 3,
            "fillColor": "#DCE6F1",
            "fillOpacity": 0.08
        },
        tooltip="Colombo Municipal Council Study Area"
    ).add_to(boundary_layer)

    boundary_layer.add_to(m)

    # --------------------------------------------------
    # Road network
    # --------------------------------------------------

    road_layer = folium.FeatureGroup(
        name="Colombo Road Network",
        show=True
    )

    folium.GeoJson(
        roads,
        name="Road Edges",
        style_function=lambda feature: {
            "color": "#777777",
            "weight": 0.7,
            "opacity": 0.55
        }
    ).add_to(road_layer)

    road_layer.add_to(m)

    # --------------------------------------------------
    # Demand proxy nodes
    # --------------------------------------------------

    demand_layer = folium.FeatureGroup(
        name="Non-candidate OSM Demand Proxy Nodes",
        show=True
    )

    for row in demand.itertuples():

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=2,
            color="#555555",
            weight=0.5,
            fill=True,
            fill_color="#777777",
            fill_opacity=0.55,
            tooltip=f"OSM demand proxy weight: {row.demand_weight}"
        ).add_to(demand_layer)

    demand_layer.add_to(m)

    # --------------------------------------------------
    # Facility separation scenarios
    # --------------------------------------------------

    for separation, color in COLORS.items():

        scenario = scenario_data[
            scenario_data["minimum_separation_km"].round(2)
            == round(separation, 2)
        ]

        if scenario.empty:
            print(f"Skipping missing scenario: {separation} km")
            continue

        if len(scenario) != p_value:
            raise ValueError(
                f"Expected {p_value} facilities but found "
                f"{len(scenario)} for separation {separation} km"
            )

        facility_layer = folium.FeatureGroup(
            name=f"Facilities - {separation:.2f} km separation",
            show=(separation == 0.0)
        )

        for row in scenario.itertuples():

            popup = (
                f"<b>Selected Facility</b><br>"
                f"Candidate ID: {row.candidate_id}<br>"
                f"Node ID: {row.node_id}<br>"
                f"Number of facilities: {p_value}<br>"
                f"Minimum separation: {separation:.2f} km<br>"
                f"Candidate source: {row.candidate_sources}"
            )

            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=8,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                tooltip=f"Facility {row.candidate_id}",
                popup=folium.Popup(popup, max_width=300)
            ).add_to(facility_layer)

        facility_layer.add_to(m)

        print(
            f"Separation {separation:.2f} km: "
            f"{len(scenario)} facilities"
        )

    # --------------------------------------------------
    # Layer controls and map extent
    # --------------------------------------------------

    folium.LayerControl(collapsed=False).add_to(m)

    m.fit_bounds([
        [south, west],
        [north, east]
    ])

    title = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 12px 16px;
        border: 2px solid #17365D;
        border-radius: 5px;
        font-size: 15px;
        font-weight: bold;">
        Colombo EV Charging Facility Optimization - P={p_value}
    </div>
    """

    m.get_root().html.add_child(folium.Element(title))

    # --------------------------------------------------
    # Save map
    # --------------------------------------------------

    output_file = OUTPUT_DIR / (
        f"mclp_spatial_comparison_p{p_value}.html"
    )

    m.save(str(output_file))

    print("Saved:", output_file)

print("\nBoth offline spatial maps generated successfully.")
