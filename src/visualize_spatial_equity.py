
from pathlib import Path
import json

import numpy as np
import pandas as pd
import folium


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = (
    ROOT / "data/raw/road_network/colombo_study_boundary.geojson"
)

ROAD_FILE = (
    ROOT / "data/raw/road_network/colombo_road_edges.geojson"
)

DEMAND_FILE = (
    ROOT / "data/processed/demand/non_candidate_demand_nodes.csv"
)

DISTANCE_FILE = (
    ROOT / "data/processed/distance/non_candidate_demand_candidate_distances_km.csv"
)

SELECTION_FILE = (
    ROOT / "outputs/optimization/mclp_lexicographic_selected_facilities.csv"
)

OUTPUT_DIR = ROOT / "outputs/visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SCENARIOS = [
    "baseline",
    "combined_quality",
]

P_VALUES = [5, 10]

COLORS = {
    "within_1km": "#16a34a",
    "within_2km": "#eab308",
    "beyond_2km": "#f97316",
    "unreachable": "#991b1b",
}

SCENARIO_COLORS = {
    "baseline": "#2563eb",
    "combined_quality": "#7c3aed",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\n========== LOADING SPATIAL DATA ==========")

    required_files = [
        BOUNDARY_FILE,
        ROAD_FILE,
        DEMAND_FILE,
        DISTANCE_FILE,
        SELECTION_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

        print("Found:", file_path)

    with open(BOUNDARY_FILE, encoding="utf-8") as f:
        boundary = json.load(f)

    with open(ROAD_FILE, encoding="utf-8") as f:
        roads = json.load(f)

    demand = pd.read_csv(DEMAND_FILE)

    distance = pd.read_csv(
        DISTANCE_FILE,
        index_col=0
    )

    selections = pd.read_csv(
        SELECTION_FILE,
        dtype={"candidate_id": str}
    )

    # Normalize identifiers.
    demand["node_id"] = demand["node_id"].astype(str)

    distance.index = distance.index.astype(str)
    distance.columns = distance.columns.astype(str)

    selections["candidate_id"] = (
        selections["candidate_id"].astype(str)
    )

    selections["scenario"] = (
        selections["scenario"].astype(str)
    )

    selections["p"] = pd.to_numeric(
        selections["p"],
        errors="raise"
    ).astype(int)

    # Validate demand coordinates.
    demand = demand.dropna(
        subset=["longitude", "latitude", "demand_weight"]
    ).copy()

    demand["longitude"] = pd.to_numeric(
        demand["longitude"],
        errors="raise"
    )

    demand["latitude"] = pd.to_numeric(
        demand["latitude"],
        errors="raise"
    )

    demand["demand_weight"] = pd.to_numeric(
        demand["demand_weight"],
        errors="raise"
    )

    # Ensure matrix rows correspond to demand records.
    if demand["node_id"].duplicated().any():
        raise ValueError("Duplicate demand node IDs found.")

    if not demand["node_id"].tolist() == distance.index.tolist():
        raise ValueError(
            "Demand node order does not match distance matrix."
        )

    if demand[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Missing demand coordinates found.")

    if not np.isfinite(
        demand[["longitude", "latitude"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid demand coordinates found.")

    print("\nBoundary features:", len(boundary["features"]))
    print("Road features:", len(roads["features"]))
    print("Demand proxy nodes:", len(demand))
    print("Distance matrix:", distance.shape)
    print("Facility selection records:", len(selections))

    return boundary, roads, demand, distance, selections


# ============================================================
# 4. CREATE BASE MAP
# ============================================================

def create_base_map(boundary):

    properties = boundary["features"][0]["properties"]

    west = float(properties["bbox_west"])
    south = float(properties["bbox_south"])
    east = float(properties["bbox_east"])
    north = float(properties["bbox_north"])

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None,
        control_scale=True
    )

    # Study boundary.
    folium.GeoJson(
        boundary,
        name="Colombo Study Area Boundary",
        style_function=lambda feature: {
            "color": "#17365D",
            "weight": 3,
            "fillColor": "#DCE6F1",
            "fillOpacity": 0.06,
        },
        tooltip="Colombo Municipal Council Study Area"
    ).add_to(m)

    m.fit_bounds([
        [south, west],
        [north, east]
    ])

    return m


# ============================================================
# 5. ADD ROAD NETWORK
# ============================================================

def add_road_network(m, roads):

    road_layer = folium.FeatureGroup(
        name="Colombo Road Network",
        show=True
    )

    folium.GeoJson(
        roads,
        name="Road Network",
        style_function=lambda feature: {
            "color": "#777777",
            "weight": 0.7,
            "opacity": 0.5,
        }
    ).add_to(road_layer)

    road_layer.add_to(m)


# ============================================================
# 6. CALCULATE NEAREST FACILITY DISTANCE
# ============================================================

def calculate_nearest_distances(
    scenario,
    p,
    demand,
    distance,
    selections
):

    selected = selections[
        (selections["scenario"] == scenario)
        & (selections["p"] == p)
    ].copy()

    if len(selected) != p:
        raise ValueError(
            f"{scenario}, P={p}: expected {p} facilities, "
            f"found {len(selected)}."
        )

    if selected["candidate_id"].duplicated().any():
        raise ValueError(
            f"Duplicate selected facilities in {scenario}, P={p}."
        )

    candidate_ids = selected["candidate_id"].tolist()

    missing_candidates = [
        candidate
        for candidate in candidate_ids
        if candidate not in distance.columns
    ]

    if missing_candidates:
        raise ValueError(
            f"Missing distance columns: {missing_candidates}"
        )

    selected_distances = distance[
        candidate_ids
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    # Ignore NaN and infinite values when finding
    # the nearest reachable selected facility.
    finite_mask = np.isfinite(selected_distances)

    nearest = np.full(
        len(demand),
        np.inf,
        dtype=float
    )

    reachable = finite_mask.any(axis=1)

    nearest[reachable] = np.min(
        np.where(
            finite_mask,
            selected_distances,
            np.inf
        ),
        axis=1
    )[reachable]

    print(
        f"{scenario}, P={p}: "
        f"{len(candidate_ids)} facilities, "
        f"{reachable.sum()} reachable demand nodes, "
        f"{(~reachable).sum()} unreachable demand nodes"
    )

    return nearest, selected


# ============================================================
# 7. ADD DEMAND ACCESSIBILITY LAYERS
# ============================================================

def add_demand_layers(
    m,
    demand,
    nearest,
    scenario,
    show
):

    scenario_title = scenario.replace("_", " ").title()

    categories = [
        (
            "within_1km",
            "Within 1 km",
            lambda d: np.isfinite(d) & (d <= 1.0),
        ),
        (
            "within_2km",
            "Between 1 and 2 km",
            lambda d: np.isfinite(d) & (d > 1.0) & (d <= 2.0),
        ),
        (
            "beyond_2km",
            "Beyond 2 km",
            lambda d: np.isfinite(d) & (d > 2.0),
        ),
        (
            "unreachable",
            "Unreachable through road network",
            lambda d: ~np.isfinite(d),
        ),
    ]

    for key, label, condition in categories:

        mask = condition(nearest)

        subset = demand.loc[mask].copy()

        layer = folium.FeatureGroup(
            name=f"{scenario_title} - {label}",
            show=show
        )

        for index, row in subset.iterrows():

            distance_value = nearest[index]

            if np.isfinite(distance_value):
                distance_text = (
                    f"{distance_value:.3f} km"
                )
            else:
                distance_text = "Unreachable"

            popup = (
                f"<b>OSM Demand Proxy Node</b><br>"
                f"Scenario: {scenario_title}<br>"
                f"Node ID: {row['node_id']}<br>"
                f"Proxy weight: {row['demand_weight']}<br>"
                f"Proxy count: {row['proxy_count']}<br>"
                f"Nearest facility distance: {distance_text}"
            )

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=4,
                color=COLORS[key],
                weight=1,
                fill=True,
                fill_color=COLORS[key],
                fill_opacity=0.75,
                tooltip=(
                    f"{row['node_id']} | "
                    f"Weight: {row['demand_weight']} | "
                    f"{distance_text}"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=300
                )
            ).add_to(layer)

        layer.add_to(m)

        print(
            f"  {scenario_title} - {label}: "
            f"{len(subset)} nodes"
        )


# ============================================================
# 8. ADD SELECTED FACILITY LAYER
# ============================================================

def add_facility_layer(
    m,
    selected,
    scenario,
    p,
    show
):

    scenario_title = scenario.replace("_", " ").title()

    color = SCENARIO_COLORS[scenario]

    layer = folium.FeatureGroup(
        name=f"{scenario_title} - Selected Facilities",
        show=show
    )

    for row in selected.itertuples():

        popup = (
            f"<b>Selected EV Charging Facility</b><br>"
            f"Scenario: {scenario_title}<br>"
            f"Facility: {row.candidate_id}<br>"
            f"P value: {p}<br>"
            f"Mean snap distance: "
            f"{row.mean_snap_distance_m:.2f} m<br>"
            f"Maximum snap distance: "
            f"{row.max_snap_distance_m:.2f} m"
        )

        folium.Marker(
            location=[
                row.latitude,
                row.longitude
            ],
            tooltip=f"Facility {row.candidate_id}",
            popup=folium.Popup(
                popup,
                max_width=300
            ),
            icon=folium.Icon(
                color="blue" if scenario == "baseline" else "purple",
                icon="bolt",
                prefix="fa"
            )
        ).add_to(layer)

    layer.add_to(m)


# ============================================================
# 9. ADD MAP TITLE
# ============================================================

def add_title(m, p):

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
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
        Colombo EV Charging Spatial Equity Analysis - P={p}
        <div style="
            font-size: 11px;
            font-weight: normal;
            margin-top: 4px;
        ">
            OSM-based demand proxy | Road-network accessibility
        </div>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(title)
    )


# ============================================================
# 10. ADD LEGEND
# ============================================================

def add_legend(m):

    legend = """
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        width: 245px;
        background: white;
        border: 1px solid #888;
        z-index: 9999;
        padding: 12px;
        font-size: 12px;
        border-radius: 5px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
        <b>Spatial Accessibility Legend</b><br><br>

        <span style="color:#16a34a;">&#9679;</span>
        Within 1 km<br>

        <span style="color:#eab308;">&#9679;</span>
        Between 1 and 2 km<br>

        <span style="color:#f97316;">&#9679;</span>
        Beyond 2 km<br>

        <span style="color:#991b1b;">&#9679;</span>
        Unreachable through road network<br><br>

        <span style="color:#2563eb;">&#9679;</span>
        Baseline facilities<br>

        <span style="color:#7c3aed;">&#9679;</span>
        Combined-quality facilities<br><br>

        <small>
        Distances are road-network distances to
        the nearest selected facility.
        </small>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend)
    )


# ============================================================
# 11. GENERATE P5 AND P10 MAPS
# ============================================================

def generate_map(
    p,
    boundary,
    roads,
    demand,
    distance,
    selections
):

    print("\n" + "=" * 60)
    print(f"GENERATING SPATIAL EQUITY MAP - P={p}")
    print("=" * 60)

    m = create_base_map(boundary)

    add_road_network(m, roads)

    for scenario in SCENARIOS:

        nearest, selected = calculate_nearest_distances(
            scenario=scenario,
            p=p,
            demand=demand,
            distance=distance,
            selections=selections
        )

        # Show combined-quality layers by default.
        # Baseline layers remain available through
        # the layer control.
        show = scenario == "combined_quality"

        print(f"\nAdding {scenario} demand layers...")

        add_demand_layers(
            m=m,
            demand=demand,
            nearest=nearest,
            scenario=scenario,
            show=show
        )

        add_facility_layer(
            m=m,
            selected=selected,
            scenario=scenario,
            p=p,
            show=show
        )

    add_title(m, p)

    add_legend(m)

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    output_file = (
        OUTPUT_DIR
        / f"spatial_equity_comparison_p{p}.html"
    )

    m.save(str(output_file))

    print("\nMap saved successfully:")
    print(output_file)


# ============================================================
# 12. MAIN
# ============================================================

def main():

    boundary, roads, demand, distance, selections = load_data()

    for p in P_VALUES:

        generate_map(
            p=p,
            boundary=boundary,
            roads=roads,
            demand=demand,
            distance=distance,
            selections=selections
        )

    print("\n" + "=" * 60)
    print("SPATIAL EQUITY VISUALIZATION COMPLETED")
    print("=" * 60)

    print("\nGenerated maps:")

    for p in P_VALUES:

        print(
            OUTPUT_DIR
            / f"spatial_equity_comparison_p{p}.html"
        )


if __name__ == "__main__":
    main()
