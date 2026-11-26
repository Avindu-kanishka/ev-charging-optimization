from pathlib import Path

import pandas as pd
from pyproj import Transformer
from sklearn.model_selection import GroupKFold

DATA_PATH = Path("data/processed/ml/spatial_demand_proxy_dataset.csv")
OUTPUT_PATH = Path("data/processed/ml/spatial_cv_assignments.csv")

df = pd.read_csv(DATA_PATH)

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32644",
    always_xy=True
)

df["x"], df["y"] = transformer.transform(
    df["longitude"].to_numpy(),
    df["latitude"].to_numpy()
)

BLOCK_SIZE_M = 500

df["block_x"] = (df["x"] // BLOCK_SIZE_M).astype(int)
df["block_y"] = (df["y"] // BLOCK_SIZE_M).astype(int)

df["spatial_block"] = (
    df["block_x"].astype(str) + "_" +
    df["block_y"].astype(str)
)

df["cv_fold"] = -1

groups = df["spatial_block"]
splitter = GroupKFold(n_splits=5)

for fold, (_, test_idx) in enumerate(
    splitter.split(df, groups=groups)
):
    df.loc[test_idx, "cv_fold"] = fold

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df[
    [
        "node_id",
        "spatial_block",
        "cv_fold",
        "demand_proxy_count"
    ]
].to_csv(OUTPUT_PATH, index=False)

print("SPATIAL CROSS-VALIDATION ASSIGNMENTS CREATED")
print("Rows:", len(df))
print("Unique spatial blocks:", df["spatial_block"].nunique())
print("Folds:", df["cv_fold"].nunique())

print("\nFold diagnostics:")

for fold in sorted(df["cv_fold"].unique()):
    part = df[df["cv_fold"] == fold]

    print(
        f"Fold {fold}: "
        f"nodes={len(part)}, "
        f"blocks={part['spatial_block'].nunique()}, "
        f"positive_nodes={(part['demand_proxy_count'] > 0).sum()}, "
        f"proxy_count={part['demand_proxy_count'].sum()}"
    )

print("\nOutput:", OUTPUT_PATH)
