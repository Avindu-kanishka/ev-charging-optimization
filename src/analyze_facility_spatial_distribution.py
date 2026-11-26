from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE
    / "outputs/optimization/"
    / "mclp_facility_separation_selected_locations.csv"
)

OUTPUT_DIR = BASE / "outputs/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "facility_spatial_distribution_metrics.csv"
)

SEPARATION_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5]
P_VALUES = [5, 10]

EARTH_RADIUS_KM = 6371.0088

# --------------------------------------------------
# 2. Haversine distance function
# --------------------------------------------------

def haversine_matrix(latitudes, longitudes):
    """
    Calculate pairwise great-circle distances in kilometres.
    """

    lat = np.radians(np.asarray(latitudes, dtype=float))
    lon = np.radians(np.asarray(longitudes, dtype=float))

    lat1 = lat[:, None]
    lat2 = lat[None, :]

    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# --------------------------------------------------
# 3. Load selected facilities
# --------------------------------------------------

print("Loading selected facility configurations...")

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "p",
    "minimum_separation_km",
    "candidate_id",
    "longitude",
    "latitude",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(
    subset=[
        "p",
        "minimum_separation_km",
        "candidate_id",
        "longitude",
        "latitude",
    ]
)

print("Total facility records:", len(df))

# --------------------------------------------------
# 4. Calculate spatial distribution metrics
# --------------------------------------------------

results = []

for p_value in P_VALUES:

    for separation in SEPARATION_LEVELS:

        scenario = df[
            (df["p"] == p_value)
            & (
                np.isclose(
                    df["minimum_separation_km"],
                    separation
                )
            )
        ].copy()

        if len(scenario) != p_value:
            raise ValueError(
                f"Expected {p_value} facilities for "
                f"P={p_value}, separation={separation}. "
                f"Found {len(scenario)}."
            )

        if scenario["candidate_id"].nunique() != p_value:
            raise ValueError(
                f"Duplicate candidate IDs in "
                f"P={p_value}, separation={separation}"
            )

        distances = haversine_matrix(
            scenario["latitude"].to_numpy(),
            scenario["longitude"].to_numpy()
        )

        # Extract upper triangular pairwise distances
        pairwise = distances[
            np.triu_indices(p_value, k=1)
        ]

        # Geographic centroid approximation
        mean_lat = scenario["latitude"].mean()
        mean_lon = scenario["longitude"].mean()

        centroid_distances = haversine_matrix(
            np.append(
                scenario["latitude"].to_numpy(),
                mean_lat
            ),
            np.append(
                scenario["longitude"].to_numpy(),
                mean_lon
            )
        )[:-1, -1]

        result = {
            "p": p_value,
            "minimum_separation_km": separation,
            "number_of_facilities": p_value,
            "number_of_pairs": len(pairwise),
            "minimum_pairwise_distance_km": pairwise.min(),
            "mean_pairwise_distance_km": pairwise.mean(),
            "median_pairwise_distance_km": np.median(pairwise),
            "maximum_pairwise_distance_km": pairwise.max(),
            "std_pairwise_distance_km": pairwise.std(ddof=0),
            "mean_distance_from_centroid_km": (
                centroid_distances.mean()
            ),
        }

        results.append(result)

        print(
            f"P={p_value}, separation={separation:.2f} km | "
            f"Min={pairwise.min():.3f} km | "
            f"Mean={pairwise.mean():.3f} km | "
            f"Max={pairwise.max():.3f} km"
        )

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["p", "minimum_separation_km"]
)

results_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. Display summary
# --------------------------------------------------

print("\n" + "=" * 85)
print("FACILITY SPATIAL DISTRIBUTION ANALYSIS")
print("=" * 85)

display_columns = [
    "p",
    "minimum_separation_km",
    "minimum_pairwise_distance_km",
    "mean_pairwise_distance_km",
    "maximum_pairwise_distance_km",
    "mean_distance_from_centroid_km",
]

print(
    results_df[display_columns].round(3).to_string(
        index=False
    )
)

print("\nResults saved to:", OUTPUT_FILE)
print("\nSpatial distribution analysis completed successfully.")
