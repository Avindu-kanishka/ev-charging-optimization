from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "outputs/analysis/facility_spatial_distribution_metrics.csv"
OUTPUT = BASE / "outputs/analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

metrics = {
    "minimum_pairwise_distance_km": (
        "Minimum Pairwise Facility Distance",
        "minimum_pairwise_distance.png"
    ),
    "mean_pairwise_distance_km": (
        "Mean Pairwise Facility Distance",
        "mean_pairwise_distance.png"
    ),
    "mean_distance_from_centroid_km": (
        "Mean Distance from Facility Centroid",
        "mean_distance_from_centroid.png"
    )
}

for metric, (title, filename) in metrics.items():

    plt.figure(figsize=(9, 5))

    for p in sorted(df["p"].unique()):
        subset = df[df["p"] == p].sort_values(
            "minimum_separation_km"
        )

        plt.plot(
            subset["minimum_separation_km"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"P = {p}"
        )

    plt.title(title)
    plt.xlabel("Minimum Facility Separation Constraint (km)")
    plt.ylabel("Geographical Distance (km)")
    plt.xticks([0, 0.25, 0.5, 1.0, 1.5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = OUTPUT / filename
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)

print("\nAll spatial distribution charts generated.")
