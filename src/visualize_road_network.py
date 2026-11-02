
from pathlib import Path

import osmnx as ox
import folium

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = (
    BASE_DIR / "data" / "raw" / "road_network"
    / "colombo_road_network.graphml"
)

MAP_DIR = BASE_DIR / "outputs" / "maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading road network...")
    G = ox.load_graphml(GRAPH_PATH)

    nodes, edges = ox.graph_to_gdfs(G)

    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()

    # No external basemap tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None
    )

    # Add roads
    folium.GeoJson(
        edges.to_json(),
        name="Colombo Road Network",
        style_function=lambda feature: {
            "color": "#2563eb",
            "weight": 1.3,
            "opacity": 0.8
        }
    ).add_to(m)

    # Add road network nodes
    folium.GeoJson(
        nodes.to_json(),
        name="Road Nodes",
        marker=folium.CircleMarker(
            radius=1,
            color="red",
            fill=True,
            fill_opacity=0.5
        )
    ).add_to(m)

    # Fit map to network extent
    minx, miny, maxx, maxy = edges.total_bounds

    m.fit_bounds([
        [miny, minx],
        [maxy, maxx]
    ])

    folium.LayerControl().add_to(m)

    output_path = MAP_DIR / "colombo_road_network.html"
    m.save(str(output_path))

    print("Map saved:", output_path)


if __name__ == "__main__":
    main()
