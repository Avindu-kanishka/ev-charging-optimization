
import requests

query = """
[out:json][timeout:60];
relation(16132);
out tags;
"""

url = "https://overpass.kumi.systems/api/interpreter"

response = requests.post(
    url,
    data={"data": query},
    headers={"User-Agent": "EVChargingResearch/1.0"},
    timeout=90
)

response.raise_for_status()

data = response.json()

for relation in data["elements"]:
    tags = relation.get("tags", {})

    print("OSM Relation ID:", relation["id"])
    print("Name:", tags.get("name"))
    print("Administrative level:", tags.get("admin_level"))
    print("Boundary:", tags.get("boundary"))
    print("All tags:", tags)
