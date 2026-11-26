from pathlib import Path
import json
import folium
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

ROAD_FILE = BASE / "data/raw/road_network/colombo_road_edges.geojson"
NODE_FILE = BASE / "data/raw/road_network/colombo_road_nodes.geojson"
CANDIDATE_FILE = BASE / "data/processed/candidates/candidate_locations.csv"

OUTPUT_DIR = BASE / "outputs/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_DEGREES = 0.006

print("Loading road network and candidate data...")

with open(ROAD_FILE, encoding="utf-8") as f:
    roads = json.load(f)

with open(NODE_FILE, encoding="utf-8") as f:
    nodes = json.load(f)

candidates = pd.read_csv(CANDIDATE_FILE)

node_coords = {}
for feature in nodes["features"]:
    props = feature["properties"]
    node_id = str(props.get("osmid", props.get("id")))
    node_coords[node_id] = feature["geometry"]["coordinates"]

targets = ["C0195", "C0198"]

for candidate_id in targets:
    candidate = candidates[
        candidates["candidate_id"] == candidate_id
    ].iloc[0]

    node_id = str(candidate["node_id"])
    lon = float(candidate["longitude"])
    lat = float(candidate["latitude"])

    west = lon - RADIUS_DEGREES
    east = lon + RADIUS_DEGREES
    south = lat - RADIUS_DEGREES
    north = lat + RADIUS_DEGREES

    print(f"\nGenerating map for {candidate_id}...")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles=None,
        control_scale=True
    )

    # Road edges near the candidate
    one_way_layer = folium.FeatureGroup(
        name="One-way road edges",
        show=True
    )

    other_road_layer = folium.FeatureGroup(
        name="Non-one-way road edges",
        show=True
    )

    incident_layer = folium.FeatureGroup(
        name="Edges directly connected to candidate",
        show=True
    )

    nearby_nodes_layer = folium.FeatureGroup(
        name="Nearby road nodes",
        show=True
    )

    for feature in roads["features"]:
        geometry = feature["geometry"]
        props = feature["properties"]

        if geometry["type"] != "LineString":
            continue

        coordinates = geometry["coordinates"]

        inside = any(
            west <= point[0] <= east and
            south <= point[1] <= north
            for point in coordinates
        )

        if not inside:
            continue

        u = str(props.get("u"))
        v = str(props.get("v"))

        is_incident = node_id in (u, v)
        is_oneway = props.get("oneway") is True

        highway = props.get("highway")
        highway_label = (
            ", ".join(highway)
            if isinstance(highway, list)
            else str(highway)
        )

        tooltip = (
            f"Road: {highway_label} | "
            f"One-way: {is_oneway} | "
            f"Length: {props.get('length', 'N/A')} m"
        )

        if is_incident:
            folium.PolyLine(
                locations=[
                    [point[1], point[0]]
                    for point in coordinates
                ],
                color="#ff0000",
                weight=6,
                opacity=1,
                tooltip=tooltip
            ).add_to(incident_layer)

        elif is_oneway:
            folium.PolyLine(
                locations=[
                    [point[1], point[0]]
                    for point in coordinates
                ],
                color="#e67e22",
                weight=3,
                opacity=0.85,
                tooltip=tooltip
            ).add_to(one_way_layer)

        else:
            folium.PolyLine(
                locations=[
                    [point[1], point[0]]
                    for point in coordinates
                ],
                color="#52616b",
                weight=2,
                opacity=0.7,
                tooltip=tooltip
            ).add_to(other_road_layer)

    for road_layer in [
        one_way_layer,
        other_road_layer,
        incident_layer
    ]:
        road_layer.add_to(m)

    # Nearby road nodes
    for nearby_id, coordinates in node_coords.items():
        nlon, nlat = coordinates

        if west <= nlon <= east and south <= nlat <= north:
            folium.CircleMarker(
                location=[nlat, nlon],
                radius=3,
                color="#333333",
                weight=1,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.9,
                tooltip=f"Road node: {nearby_id}"
            ).add_to(nearby_nodes_layer)

    nearby_nodes_layer.add_to(m)

    # Candidate node
    folium.Marker(
        location=[lat, lon],
        tooltip=f"Candidate {candidate_id}",
        popup=folium.Popup(
            f"""
            <b>Candidate:</b> {candidate_id}<br>
            <b>Node:</b> {node_id}<br>
            <b>Source:</b> {candidate['candidate_sources']}<br>
            <b>Mean snap distance:</b>
            {candidate['mean_snap_distance_m']:.2f} m<br>
            <b>Maximum snap distance:</b>
            {candidate['max_snap_distance_m']:.2f} m<br>
            <b>Over 100 m:</b> {candidate['snap_over_100m']}
            """,
            max_width=300
        ),
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

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
        padding: 12px;
        border: 2px solid #17365D;
        border-radius: 5px;
        font-size: 15px;
        font-weight: bold;">
        Road Connectivity Diagnostic - {candidate_id}
    </div>
    """

    m.get_root().html.add_child(folium.Element(title))

    output_file = OUTPUT_DIR / (
        f"connectivity_diagnostic_{candidate_id}.html"
    )

    m.save(str(output_file))
    print("Saved:", output_file)

print("\nBoth diagnostic maps generated.")
