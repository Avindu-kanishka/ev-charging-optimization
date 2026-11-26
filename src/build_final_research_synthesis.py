from pathlib import Path
import pandas as pd

out = Path("outputs/analysis")
out.mkdir(parents=True, exist_ok=True)

lex = pd.read_csv("outputs/optimization/mclp_lexicographic_optimization_results.csv")
equity = pd.read_csv("outputs/analysis/spatial_equity_distance_metrics.csv")
access = pd.read_csv("outputs/analysis/spatial_accessibility_coverage_summary.csv")
pmedian = pd.read_csv("outputs/analysis/comparative_pmedian_cross_evaluation.csv")
mclp = pd.read_csv("outputs/analysis/comparative_mclp_evaluation.csv")

combined = equity[
    (equity["scenario"] == "combined_quality") &
    (equity["p"] == 10)
].iloc[0]

baseline = equity[
    (equity["scenario"] == "baseline") &
    (equity["p"] == 10)
].iloc[0]

selected = pd.read_csv(
    "outputs/optimization/mclp_lexicographic_selected_facilities.csv"
)
sites = selected[
    (selected["scenario"] == "combined_quality") &
    (selected["p"] == 10)
]["candidate_id"].tolist()

summary = pd.DataFrame([
    ["Combined-quality P=10", "Coverage within 3 km (%)", combined["reachable_proxy_weight"] / combined["total_proxy_weight"] * 100],
    ["Combined-quality P=10", "Weighted mean distance (km)", combined["weighted_mean_distance_km"]],
    ["Combined-quality P=10", "Weighted P90 distance (km)", combined["weighted_p90_distance_km"]],
    ["Combined-quality P=10", "Proxy weight beyond 1 km (%)", combined["proxy_weight_beyond_1km_pct"]],
    ["Combined-quality P=10", "Proxy weight beyond 2 km (%)", combined["proxy_weight_beyond_2km_pct"]],
    ["Combined-quality P=10", "Unreachable demand nodes", combined["unreachable_demand_nodes"]],
    ["Baseline P=10", "Weighted mean distance (km)", baseline["weighted_mean_distance_km"]],
    ["Baseline P=10", "Weighted P90 distance (km)", baseline["weighted_p90_distance_km"]],
    ["Baseline P=10", "Proxy weight beyond 2 km (%)", baseline["proxy_weight_beyond_2km_pct"]],
    ["Baseline P=10", "Unreachable demand nodes", baseline["unreachable_demand_nodes"]],
], columns=["configuration", "metric", "value"])

summary.to_csv(out / "final_research_results_summary.csv", index=False)

report = f"""# EV Charging Station Location Optimization
## Final Research Results Synthesis

### 1. Research scope

This study investigates EV charging station location optimization in the Colombo Municipal Council area using road-network analysis, graph theory, and mathematical optimization.

Demand is represented by an **OSM-based urban-activity demand proxy**. It is not observed EV ownership, charging sessions, or measured EV charging demand.

### 2. Recommended configuration

The recommended research configuration is the combined-quality P=10 lexicographic MCLP solution.

Selected candidate IDs:

{", ".join(sites)}

- Coverage radius: 3 km
- Minimum facility separation: 0.5 km
- Eligible candidates: 256
- Covered proxy weight: {combined['reachable_proxy_weight']:.0f} / {combined['total_proxy_weight']:.0f}
- Proxy-weight coverage: {combined['reachable_proxy_weight'] / combined['total_proxy_weight'] * 100:.3f}%
- Weighted mean distance: {combined['weighted_mean_distance_km']:.3f} km
- Weighted P90 distance: {combined['weighted_p90_distance_km']:.3f} km
- Unreachable demand nodes: {int(combined['unreachable_demand_nodes'])}

### 3. Baseline comparison

| Measure | Baseline P=10 | Combined-quality P=10 |
|---|---:|---:|
| Weighted mean distance (km) | {baseline['weighted_mean_distance_km']:.3f} | {combined['weighted_mean_distance_km']:.3f} |
| Weighted P90 distance (km) | {baseline['weighted_p90_distance_km']:.3f} | {combined['weighted_p90_distance_km']:.3f} |
| Proxy weight beyond 2 km (%) | {baseline['proxy_weight_beyond_2km_pct']:.2f} | {combined['proxy_weight_beyond_2km_pct']:.2f} |
| Unreachable demand nodes | {int(baseline['unreachable_demand_nodes'])} | {int(combined['unreachable_demand_nodes'])} |
| Distance Gini | {baseline['distance_gini']:.3f} | {combined['distance_gini']:.3f} |

The combined-quality solution improves weighted mean and P90 access distances and reduces the proportion of proxy weight beyond 2 km. However, it has one additional unreachable node and a slightly higher distance Gini. Therefore, the improvement is not uniform across every equity measure.

### 4. Road-network and Euclidean comparison

The comparative evaluation shows that Euclidean MCLP achieved 100% proxy-weight coverage in all tested scenarios. Road-network MCLP achieved 99.768% for P=5 at 3 km and 99.954% in the other tested scenarios.

Straight-line distances therefore provide an optimistic approximation of accessibility compared with road-network distances.

### 5. Methodological contributions

- Road-network representation of the study area.
- Shortest-path distance-based accessibility analysis.
- P-Median and MCLP optimization.
- Euclidean baseline comparison.
- Candidate road-network and snapping-quality assessment.
- Demand-weight, coverage-radius, and facility-separation sensitivity.
- Lexicographic optimization prioritizing coverage and then snapping quality.
- Spatial accessibility and distance-equity evaluation.

### 6. Limitations

1. OSM activity locations are proxies, not observed EV charging demand.
2. OSM completeness and positional accuracy may vary.
3. Candidate locations are constrained by the collected candidate dataset.
4. Road-network distances do not necessarily represent real-time traffic or travel time.
5. Results depend on the selected coverage radius, facility count, and quality filters.
6. Optimization solver time limits and gap tolerances must be reported; a solver status alone does not prove global optimality.
7. Financial feasibility, electricity-grid capacity, land ownership, and actual EV adoption were not established by this spatial model.

### 7. Conclusion

The results demonstrate that incorporating road-network accessibility and candidate quality can materially change the spatial optimization outcome. The combined-quality P=10 solution provides a strong balance of high proxy-weight coverage, improved average access distance, and better candidate snapping quality, although a small number of demand nodes remain unreachable and some equity measures do not improve.

The selected locations should be interpreted as research-generated candidate sites requiring field verification, actual EV-demand data, land assessment, and electricity-grid feasibility analysis before implementation.

---
Generated from the project's existing optimization and analysis outputs.
"""

(out / "final_research_synthesis.md").write_text(report, encoding="utf-8")

print("FINAL SYNTHESIS GENERATED")
print(out / "final_research_synthesis.md")
print(out / "final_research_results_summary.csv")
print("\nSelected sites:", ", ".join(sites))
print("Combined-quality coverage:",
      round(combined["reachable_proxy_weight"] /
            combined["total_proxy_weight"] * 100, 3), "%")
print("Baseline weighted mean:",
      round(baseline["weighted_mean_distance_km"], 3), "km")
print("Combined weighted mean:",
      round(combined["weighted_mean_distance_km"], 3), "km")
