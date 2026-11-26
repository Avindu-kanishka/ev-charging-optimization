# EV Charging Station Location Optimization
## Final Research Results Synthesis

### 1. Research scope

This study investigates EV charging station location optimization in the Colombo Municipal Council area using road-network analysis, graph theory, and mathematical optimization.

Demand is represented by an **OSM-based urban-activity demand proxy**. It is not observed EV ownership, charging sessions, or measured EV charging demand.

### 2. Recommended configuration

The recommended research configuration is the combined-quality P=10 lexicographic MCLP solution.

Selected candidate IDs:

C0063, C0098, C0111, C0172, C0207, C0213, C0242, C0245, C0263, C0274

- Coverage radius: 3 km
- Minimum facility separation: 0.5 km
- Eligible candidates: 256
- Covered proxy weight: 1773 / 1775
- Proxy-weight coverage: 99.887%
- Weighted mean distance: 1.282 km
- Weighted P90 distance: 2.005 km
- Unreachable demand nodes: 2

### 3. Baseline comparison

| Measure | Baseline P=10 | Combined-quality P=10 |
|---|---:|---:|
| Weighted mean distance (km) | 1.451 | 1.282 |
| Weighted P90 distance (km) | 2.213 | 2.005 |
| Proxy weight beyond 2 km (%) | 20.34 | 10.20 |
| Unreachable demand nodes | 1 | 2 |
| Distance Gini | 0.247 | 0.258 |

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
