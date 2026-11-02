def load_data():
    print("\n[1/4] Loading optimization data...")

    demand_path = (
        PROJECT_ROOT
        / "data/processed/demand/road_demand_nodes.csv"
    )

    candidate_path = (
        PROJECT_ROOT
        / "data/processed/candidates/candidate_locations.csv"
    )

    distance_path = (
        PROJECT_ROOT
        / "data/processed/distance/demand_candidate_distances_km.csv"
    )

    demand = pd.read_csv(demand_path)
    candidates = pd.read_csv(candidate_path)
    distances = pd.read_csv(distance_path, index_col=0)

    # Normalize node IDs to strings
    demand["node_id"] = demand["node_id"].astype(str)
    candidates["candidate_id"] = (
        candidates["candidate_id"].astype(str)
    )

    distances.index = distances.index.astype(str)
    distances.columns = distances.columns.astype(str)

    # Align demand rows with distance matrix
    demand = demand.set_index("node_id")

    missing_demand = distances.index.difference(demand.index)

    if len(missing_demand) > 0:
        raise ValueError(
            f"{len(missing_demand)} demand nodes are missing "
            "from the demand dataset."
        )

    demand = demand.loc[distances.index]

    # Align candidate columns with candidate dataset
    candidates = candidates.set_index("candidate_id")

    missing_candidates = distances.columns.difference(
        candidates.index
    )

    if len(missing_candidates) > 0:
        raise ValueError(
            f"{len(missing_candidates)} candidates are missing "
            "from the candidate dataset."
        )

    candidates = candidates.loc[distances.columns]

    # Distance matrix
    D = distances.to_numpy(dtype=float)

    # Demand weights
    weights = demand["demand_weight"].to_numpy(dtype=float)

    demand_ids = distances.index.tolist()
    candidate_ids = distances.columns.tolist()

    print(f"   Demand nodes: {len(demand_ids):,}")
    print(f"   Candidate locations: {len(candidate_ids):,}")
    print(f"   Distance matrix: {D.shape}")

    print(
        f"   Unreachable distance pairs: "
        f"{np.isinf(D).sum():,}"
    )

    print(
        f"   Total demand proxy weight: "
        f"{weights.sum():,.0f}"
    )

    return D, weights, demand_ids, candidate_ids, candidates

if __name__ == "__main__":
    main()
