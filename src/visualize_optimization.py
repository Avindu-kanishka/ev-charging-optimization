
from pathlib import Path

import pandas as pd
import folium
import osmnx as ox
import geopandas as gpd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    ROOT / "data/raw/road_network/colombo_road_network.graphml"
)

BOUNDARY_PATH = (
    ROOT / "data/raw/road_network/colombo_study_boundary.geojson"
)

DEMAND_PATH = (
    ROOT / "data/processed/demand/road_demand_nodes.csv"
)

SELECTED_PATH = (
    ROOT / "outputs/optimization/selected_locations.csv"
)

PMEDIAN_RESULTS = (
    ROOT / "outputs/optimization/p_median_results.csv"
)

OUTPUT_DIR = ROOT / "outputs/maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "optimization_locations.html"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading road network...")

G = ox.load_graphml(GRAPH_PATH)

nodes, edges = ox.graph_to_gdfs(G)

demand = pd.read_csv(DEMAND_PATH)
selected = pd.read_csv(SELECTED_PATH)
p_results = pd.read_csv(PMEDIAN_RESULTS)

print(f"Road nodes: {len(nodes):,}")
print(f"Road edges: {len(edges):,}")
print(f"Demand nodes: {len(demand):,}")
print(f"Selected location records: {len(selected):,}")


# ============================================================
# CREATE BASE MAP
# ============================================================

center_lat = demand["latitude"].mean()
center_lon = demand["longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="OpenStreetMap"
)


# ============================================================
# STUDY AREA BOUNDARY
# ============================================================

boundary = gpd.read_file(BOUNDARY_PATH)

folium.GeoJson(
    boundary.to_json(),
    name="Study Area Boundary",
    style_function=lambda feature: {
        "color": "#222222",
        "weight": 3,
        "fill": False
    }
).add_to(m)


# ============================================================
# ROAD NETWORK
# ============================================================

road_group = folium.FeatureGroup(
    name="Road Network",
    show=True
)

edges_wgs84 = edges.to_crs(epsg=4326)

folium.GeoJson(
    edges_wgs84.to_json(),
    style_function=lambda feature: {
        "color": "#64748b",
        "weight": 1,
        "opacity": 0.55
    },
    name="Road Edges"
).add_to(road_group)

road_group.add_to(m)


# ============================================================
# DEMAND PROXY LOCATIONS
# ============================================================

demand_group = folium.FeatureGroup(
    name="Demand Proxy Locations",
    show=True
)

for _, row in demand.iterrows():

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=2,
        color="#2563eb",
        fill=True,
        fill_opacity=0.55,
        weight=1,
        popup=(
            f"Demand node: {row['node_id']}<br>"
            f"Proxy weight: {row['demand_weight']}"
        )
    ).add_to(demand_group)

demand_group.add_to(m)


# ============================================================
# P-MEDIAN LAYERS
# ============================================================

p5_status = p_results.loc[
    p_results["scenario"] == "P=5",
    "solver_status"
]

p5_label = (
    "P-Median P=5 (Not Solved - incumbent)"
    if not p5_status.empty and p5_status.iloc[0] != "Optimal"
    else "P-Median P=5"
)

p_scenarios = [
    ("P=5", p5_label),
    ("P=10", "P-Median P=10"),
    ("P=15", "P-Median P=15")
]

p_colors = {
    "P=5": "#dc2626",
    "P=10": "#ea580c",
    "P=15": "#9333ea"
}

for scenario, layer_name in p_scenarios:

    subset = selected[
        (selected["method"] == "P-Median") &
        (selected["scenario"] == scenario)
    ]

    group = folium.FeatureGroup(
        name=layer_name,
        show=(scenario == "P=5")
    )

    for _, row in subset.iterrows():

        folium.Marker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            tooltip=f"{row['candidate_id']} - {scenario}",
            popup=(
                f"<b>P-Median</b><br>"
                f"Scenario: {scenario}<br>"
                f"Candidate: {row['candidate_id']}<br>"
                f"Source: {row['candidate_sources']}<br>"
                f"Longitude: {row['longitude']}<br>"
                f"Latitude: {row['latitude']}"
            ),
            icon=folium.Icon(
                color="red" if scenario == "P=5"
                else "orange" if scenario == "P=10"
                else "purple",
                icon="flash",
                prefix="fa"
            )
        ).add_to(group)

    group.add_to(m)


# ============================================================
# MCLP LAYERS
# ============================================================

mclp_scenarios = sorted(
    selected.loc[
        selected["method"] == "MCLP",
        "scenario"
    ].unique()
)

for scenario in mclp_scenarios:

    subset = selected[
        (selected["method"] == "MCLP") &
        (selected["scenario"] == scenario)
    ]

    group = folium.FeatureGroup(
        name=f"MCLP - {scenario}",
        show=False
    )

    for _, row in subset.iterrows():

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=7,
            color="#059669",
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.85,
            weight=2,
            tooltip=(
                f"{row['candidate_id']} - {scenario}"
            ),
            popup=(
                f"<b>MCLP</b><br>"
                f"Scenario: {scenario}<br>"
                f"Candidate: {row['candidate_id']}<br>"
                f"Source: {row['candidate_sources']}<br>"
                f"Longitude: {row['longitude']}<br>"
                f"Latitude: {row['latitude']}"
            )
        ).add_to(group)

    group.add_to(m)


# ============================================================
# LAYER CONTROL AND LEGEND
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)

legend = """
<div style="
position: fixed;
bottom: 35px;
left: 35px;
width: 230px;
background-color: white;
border: 1px solid #777;
z-index: 9999;
padding: 12px;
font-size: 13px;
">
<b>Map Legend</b><br><br>
<span style="color:#64748b;">●</span> Road network<br>
<span style="color:#2563eb;">●</span> Demand proxies<br>
<span style="color:#dc2626;">●</span> P-Median P=5<br>
<span style="color:#ea580c;">●</span> P-Median P=10<br>
<span style="color:#9333ea;">●</span> P-Median P=15<br>
<span style="color:#059669;">●</span> MCLP locations
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend)
)


# ============================================================
# SAVE MAP
# ============================================================

m.save(str(OUTPUT_PATH))

print("\n" + "=" * 55)
print("OPTIMIZATION MAP CREATED")
print("=" * 55)

print(f"Output: {OUTPUT_PATH}")
print("Open the HTML file in your browser.")
