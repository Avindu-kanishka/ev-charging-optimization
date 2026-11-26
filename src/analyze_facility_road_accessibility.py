from pathlib import Path
import json
import math

import pandas as pd
import networkx as nx

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

ROAD_FILE = BASE / "data/raw/road_network/colombo_road_edges.geojson"

SELECTED_FILE = (
    BASE
    / "outputs/optimization/"
    / "mclp_facility_separation_selected_locations.csv"
)

OUTPUT_DIR = BASE / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "facility_road_accessibility_summary.csv"
DETAIL_FILE = OUTPUT_DIR / "facility_road_pairwise_distances.csv"

EARTH_RADIUS_KM = 6371.0088

# --------------------------------------------------
# 2. Haversine distance
# --------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(
        math.sqrt(min(1.0, a))
    )

# --------------------------------------------------
# 3. Load road network
# --------------------------------------------------

print("Loading road network...")

with open(ROAD_FILE, encoding="utf-8") as f:
    road_data = json.load(f)

G = nx.MultiDiGraph()

for feature in road_data["features"]:

    props = feature["properties"]

    u = int(props["u"])
    v = int(props["v"])
    length = float(props["length"])

    if length < 0:
        raise ValueError("Negative road length detected.")

    G.add_edge(u, v, length=length)

    # Respect one-way roads.
    # Missing oneway values are treated as two-way.
    if props.get("oneway") is not True:
        G.add_edge(v, u, length=length)

print("Graph nodes:", G.number_of_nodes())
print("Graph directed edges:", G.number_of_edges())

# --------------------------------------------------
# 4. Load selected facilities
# --------------------------------------------------

selected = pd.read_csv(SELECTED_FILE)

required = {
    "p",
    "minimum_separation_km",
    "candidate_id",
    "node_id",
    "longitude",
    "latitude",
}

if not required.issubset(selected.columns):
    raise ValueError("Selected facility dataset has missing columns.")

selected["node_id"] = selected["node_id"].astype(int)

missing_nodes = set(selected["node_id"]) - set(G.nodes())

if missing_nodes:
    raise ValueError(
        f"Selected nodes missing from graph: {missing_nodes}"
    )

print("Selected facility records:", len(selected))

# --------------------------------------------------
# 5. Calculate directed shortest paths
# --------------------------------------------------

summary_results = []
detail_results = []

scenario_groups = selected.groupby(
    ["p", "minimum_separation_km"],
    sort=True
)

for (p_value, separation), scenario in scenario_groups:

    scenario = scenario.reset_index(drop=True)

    if len(scenario) != int(p_value):
        raise ValueError(
            f"Incorrect facility count for P={p_value}, "
            f"separation={separation}"
        )

    print(
        f"\nProcessing P={p_value}, "
        f"separation={separation:.2f} km"
    )

    # Calculate Dijkstra distances from every facility.
    shortest_paths = {}

    for row in scenario.itertuples():

        distances = nx.single_source_dijkstra_path_length(
            G,
            source=int(row.node_id),
            weight="length"
        )

        shortest_paths[int(row.node_id)] = distances

    road_distances = []
    straight_distances = []
    stretch_ratios = []

    reachable_count = 0
    unreachable_count = 0

    total_ordered_pairs = int(p_value) * (int(p_value) - 1)

    # --------------------------------------------------
    # 6. Evaluate every directed facility pair
    # --------------------------------------------------

    for source in scenario.itertuples():

        source_id = int(source.node_id)

        for target in scenario.itertuples():

            target_id = int(target.node_id)

            if source_id == target_id:
                continue

            straight_km = haversine_km(
                source.latitude,
                source.longitude,
                target.latitude,
                target.longitude
            )

            road_m = shortest_paths[source_id].get(target_id)

            if road_m is None:

                unreachable_count += 1

                detail_results.append({
                    "p": int(p_value),
                    "minimum_separation_km": float(separation),
                    "source_candidate_id": source.candidate_id,
                    "target_candidate_id": target.candidate_id,
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "straight_line_distance_km": straight_km,
                    "road_network_distance_km": None,
                    "stretch_ratio": None,
                    "reachable": False,
                })

                continue

            road_km = road_m / 1000.0

            reachable_count += 1

            road_distances.append(road_km)
            straight_distances.append(straight_km)

            ratio = (
                road_km / straight_km
                if straight_km > 0
                else None
            )

            if ratio is not None:
                stretch_ratios.append(ratio)

            detail_results.append({
                "p": int(p_value),
                "minimum_separation_km": float(separation),
                "source_candidate_id": source.candidate_id,
                "target_candidate_id": target.candidate_id,
                "source_node_id": source_id,
                "target_node_id": target_id,
                "straight_line_distance_km": straight_km,
                "road_network_distance_km": road_km,
                "stretch_ratio": ratio,
                "reachable": True,
            })

    # --------------------------------------------------
    # 7. Scenario summary
    # --------------------------------------------------

    if road_distances:

        summary = {
            "p": int(p_value),
            "minimum_separation_km": float(separation),
            "number_of_facilities": int(p_value),
            "total_ordered_pairs": total_ordered_pairs,
            "reachable_ordered_pairs": reachable_count,
            "unreachable_ordered_pairs": unreachable_count,
            "reachability_percentage": (
                100 * reachable_count / total_ordered_pairs
            ),
            "mean_road_distance_km": sum(road_distances) / len(road_distances),
            "median_road_distance_km": pd.Series(road_distances).median(),
            "minimum_road_distance_km": min(road_distances),
            "maximum_road_distance_km": max(road_distances),
            "mean_straight_line_distance_km": (
                sum(straight_distances) / len(straight_distances)
            ),
            "mean_stretch_ratio": (
                sum(stretch_ratios) / len(stretch_ratios)
                if stretch_ratios else None
            ),
        }

    else:

        summary = {
            "p": int(p_value),
            "minimum_separation_km": float(separation),
            "number_of_facilities": int(p_value),
            "total_ordered_pairs": total_ordered_pairs,
            "reachable_ordered_pairs": 0,
            "unreachable_ordered_pairs": unreachable_count,
            "reachability_percentage": 0,
            "mean_road_distance_km": None,
            "median_road_distance_km": None,
            "minimum_road_distance_km": None,
            "maximum_road_distance_km": None,
            "mean_straight_line_distance_km": None,
            "mean_stretch_ratio": None,
        }

    summary_results.append(summary)

    print(
        f"Reachable: {reachable_count}/{total_ordered_pairs} | "
        f"Unreachable: {unreachable_count}"
    )

    if road_distances:
        print(
            f"Mean road distance: "
            f"{summary['mean_road_distance_km']:.3f} km"
        )

# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)
detail_df = pd.DataFrame(detail_results)

summary_df.to_csv(SUMMARY_FILE, index=False)
detail_df.to_csv(DETAIL_FILE, index=False)

print("\n" + "=" * 90)
print("ROAD-NETWORK ACCESSIBILITY RESULTS")
print("=" * 90)

print(
    summary_df[
        [
            "p",
            "minimum_separation_km",
            "total_ordered_pairs",
            "reachable_ordered_pairs",
            "unreachable_ordered_pairs",
            "reachability_percentage",
            "mean_road_distance_km",
            "mean_stretch_ratio",
        ]
    ].round(3).to_string(index=False)
)

print("\nSummary saved:", SUMMARY_FILE)
print("Pairwise details saved:", DETAIL_FILE)

print("\nRoad-network accessibility analysis completed.")
