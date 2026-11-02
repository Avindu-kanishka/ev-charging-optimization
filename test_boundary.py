
import requests

url = "https://api.openstreetmap.org/api/0.6/relation/16132.json"

try:
    response = requests.get(
        url,
        headers={"User-Agent": "EVChargingResearch/1.0"},
        timeout=30
    )

    response.raise_for_status()

    data = response.json()
    relation = data["elements"][0]
    tags = relation.get("tags", {})

    print("OSM Relation ID:", relation["id"])
    print("Name:", tags.get("name"))
    print("Administrative level:", tags.get("admin_level"))
    print("Boundary:", tags.get("boundary"))
    print("\nAll tags:")

    for key, value in tags.items():
        print(f"{key}: {value}")

except requests.RequestException as error:
    print("API request failed:", error)
