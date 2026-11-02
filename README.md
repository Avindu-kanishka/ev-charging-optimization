# Graph-Based Optimization for EV Charging Station Placement in Colombo, Sri Lanka

A research-oriented Graph Theory and Geospatial Optimization project that models Colombo's urban road network and evaluates candidate locations for electric vehicle (EV) charging infrastructure.

The current implementation builds a directed road-network graph from OpenStreetMap (OSM), maps urban-activity features to road nodes as **demand proxies**, calculates demand-to-candidate shortest-path distances, and evaluates facility placement using the P-Median and Maximal Covering Location Problem (MCLP) models.

> **Research status:** This repository contains the current baseline implementation. Advanced capacity-constrained, equity-aware and multi-objective models, improved demand estimation, real-world feasibility checks and the interactive decision-support dashboard are planned extensions—not completed features.

## Table of contents

- [Research problem](#research-problem)
- [Current workflow](#current-workflow)
- [Current dataset and network](#current-dataset-and-network)
- [Optimization models](#optimization-models)
- [Baseline results](#baseline-results)
- [Repository structure](#repository-structure)
- [Technology stack](#technology-stack)
- [Run the project](#run-the-project)
- [Limitations and interpretation](#limitations-and-interpretation)
- [Planned improvements](#planned-improvements)
- [Data sources and attribution](#data-sources-and-attribution)

## Research problem

Where should a limited number of EV charging stations be located so that urban demand areas can access charging infrastructure through the road network?

The project represents the road system as a directed graph:

\[
G = (V,E)
\]

where road intersections and network points are represented by vertices and road segments by directed edges. Edge lengths are used as travel-distance weights.

Demand proxy locations and candidate charging locations are matched to road-network nodes. The resulting directed shortest-path distance matrix is used as input to facility-location optimization models.

## Current workflow

1. **Road-network collection:** Retrieve and construct the Colombo driving network using OSMnx and OpenStreetMap data.
2. **Demand-proxy collection:** Extract urban-activity features such as hospitals, clinics, schools, universities, marketplaces, fuel stations, parking, shops and offices.
3. **Candidate collection:** Identify preliminary candidate features from OSM categories including parking, fuel, marketplaces, supermarkets and malls.
4. **Map matching:** Associate demand and candidate features with nearby road-network nodes.
5. **Distance calculation:** Calculate directed demand-to-candidate shortest-path distances using Dijkstra's algorithm and road-edge length.
6. **Comparison:** Build a straight-line (Haversine) distance matrix for comparison with road-network distances.
7. **Optimization:** Run P-Median and MCLP scenarios with different facility counts and coverage radii.
8. **Visualization:** Explore the network and selected locations using interactive Folium maps.

## Current dataset and network

The current baseline data includes:

| Component | Current count / value |
|---|---:|
| Road-network nodes | 5,453 |
| Directed road-network edges | 11,778 |
| OSM urban-activity features collected | 2,158 |
| Unique mapped demand road nodes | 1,237 |
| Unique candidate road nodes | 294 |
| Demand-to-candidate distance matrix | 1,237 × 294 |
| Reachable demand-candidate pairs | 360,331 |
| Unreachable demand-candidate pairs | 3,347 |

The road-network distance matrix has a mean reachable distance of approximately **4.305 km** and a median of approximately **3.942 km**. The straight-line distance matrix has a mean of approximately **3.375 km** and a median of approximately **3.006 km**.

In this baseline, the mean road-network distance is approximately **27.6% higher** than the mean straight-line distance. These are pairwise distance summaries, not estimates of actual trip distances or travel times.

## Optimization models

### 1. P-Median

The P-Median model selects exactly \(P\) candidate facilities and assigns each assignable demand node to one selected, reachable facility while minimizing total demand-weighted road distance.

The current experiments use:

- \(P = 5\)
- \(P = 10\)
- \(P = 15\)

### 2. Maximal Covering Location Problem (MCLP)

The MCLP model selects a fixed number of facilities to maximize the demand-proxy weight covered within a specified road-network distance threshold.

The current experiments use:

- Facility counts: \(P = 5, 10, 15\)
- Coverage radii: 3 km, 5 km and 10 km

Unreachable demand-candidate pairs are excluded from assignments and coverage relationships.

## Baseline results

### Road-network P-Median

| Facilities | Solver status in saved results | Weighted average distance |
|---:|---|---:|
| 5 | Not Solved | 1.552 km* |
| 10 | Optimal | 0.996 km |
| 15 | Optimal | 0.796 km |

\*The P=5 result is **not proven optimal** and must not be presented as an optimal solution. The saved output contains an objective value, but the solver status is Not Solved.

### Road-network MCLP

| Facilities | Radius | Covered proxy weight | Coverage |
|---:|---:|---:|---:|
| 5 | 3 km | 2,153 / 2,158 | 99.77% |
| 5 | 5 km | 2,157 / 2,158 | 99.95% |
| 5 | 10 km | 2,157 / 2,158 | 99.95% |
| 10 | 3 km | 2,157 / 2,158 | 99.95% |
| 10 | 5 km | 2,157 / 2,158 | 99.95% |
| 10 | 10 km | 2,157 / 2,158 | 99.95% |
| 15 | 3 km | 2,157 / 2,158 | 99.95% |
| 15 | 5 km | 2,157 / 2,158 | 99.95% |
| 15 | 10 km | 2,157 / 2,158 | 99.95% |

These are the values recorded in the current optimization summary. The very high proxy-coverage values should be independently checked against the selected facilities and distance matrix before being interpreted as evidence of real-world EV charging coverage. Proxy weight is not observed EV demand.

## Repository structure

```text
ev-charging-optimization/
├── data/
│   ├── raw/
│   │   ├── demand/
│   │   └── road_network/
│   └── processed/
│       ├── candidates/
│       ├── demand/
│       └── distance/
├── outputs/
│   ├── analysis/
│   ├── maps/
│   └── optimization/
├── src/
│   ├── analyze_road_network.py
│   ├── build_distance_matrix.py
│   ├── build_euclidean_matrix.py
│   ├── collect_candidate_locations.py
│   ├── collect_demand_points.py
│   ├── collect_road_network.py
│   ├── inspect_boundary.py
│   ├── map_demand_to_roads.py
│   ├── solve_euclidean_optimization.py
│   ├── solve_optimization.py
│   ├── validate_demand_matching.py
│   ├── visualize_optimization.py
│   └── visualize_road_network.py
├── .gitignore
└── README.md
```

## Technology stack

- **Python** — project implementation
- **OSMnx** — street-network and OpenStreetMap feature retrieval
- **NetworkX** — directed graph representation and shortest-path analysis
- **GeoPandas / Shapely** — geospatial data processing
- **Pandas / NumPy** — data preparation and numerical computation
- **PuLP / CBC** — mathematical optimization
- **Folium** — interactive map visualization

## Run the project

### 1. Clone the repository

```bash
git clone https://github.com/Avindu-kanishka/ev-charging-optimization.git
cd ev-charging-optimization
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project dependencies

```bash
python -m pip install --upgrade pip
pip install osmnx networkx geopandas pandas numpy shapely pulp==2.9.0 folium
```

The scripts use relative paths derived from the project directory. Run them from the repository root. The data-collection scripts require an internet connection to retrieve OpenStreetMap data.

### 4. Run the existing pipeline

Run the scripts in the following order when rebuilding the data from source:

```bash
python src/collect_road_network.py
python src/inspect_boundary.py
python src/collect_demand_points.py
python src/collect_candidate_locations.py
python src/map_demand_to_roads.py
python src/validate_demand_matching.py
python src/analyze_road_network.py
python src/build_distance_matrix.py
python src/build_euclidean_matrix.py
python src/solve_optimization.py
python src/solve_euclidean_optimization.py
python src/visualize_road_network.py
python src/visualize_optimization.py
```

Some scripts download or regenerate data and may take time. Existing processed data and outputs are included in the repository for inspection. Check each script's configuration and output paths before rerunning it.

## Limitations and interpretation

- **Demand proxies are not EV demand.** OSM amenities and commercial/activity features are used as spatial indicators. They do not represent measured EV ownership, charging sessions, traffic flows or charging requests.
- **Candidate locations are preliminary.** OSM parking, fuel, marketplace, supermarket and mall features have not been verified as available parcels or approved charging sites.
- **The study boundary is provisional.** The current boundary was obtained through geocoding a Colombo place query. It should not be described as a verified official Colombo Municipal Council boundary.
- **Distance is not travel time.** Road-edge length is the current routing weight. Congestion, travel speed, turn penalties and time-dependent traffic are not modelled.
- **Network direction matters.** Distances are calculated from demand nodes to candidate nodes on a directed graph; the reverse direction is not assumed to be equal.
- **Unreachable pairs exist.** The current distance matrix contains 3,347 unreachable demand-candidate pairs. One demand node has no reachable candidate in the current baseline.
- **Solver status matters.** A time-limited or Not Solved run must not be reported as proven optimal. Interpret each scenario using its recorded solver status and solver log.
- **Feasibility is not verified.** Land ownership, electrical-grid capacity, installation cost, charger throughput, permitting and regulatory approval are not yet incorporated.

## Planned improvements

- Improve demand estimation using validated population, traffic, EV registration and mobility datasets where available.
- Add A* search, centrality and network-accessibility analysis.
- Implement capacitated P-Median and equity-aware facility-location models.
- Develop multi-objective optimization for distance, coverage, cost and equity.
- Add existing charging stations and real-world candidate-feasibility criteria.
- Perform sensitivity analysis, scenario testing and independent solution validation.
- Build a Streamlit decision-support dashboard for interactive scenario comparison.

## Data sources and attribution

- **OpenStreetMap contributors:** Road-network and urban-activity features. OpenStreetMap data is available under the Open Database License (ODbL). See [OpenStreetMap Copyright and License](https://www.openstreetmap.org/copyright).
- **OSMnx:** Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139. https://doi.org/10.1016/j.compenvurbsys.2017.05.004

If you use this project or its data, retain appropriate OpenStreetMap attribution and review the applicable ODbL requirements.

---

**Author:** Avindu Kanishka  
**Study area:** Colombo, Sri Lanka  
**Project type:** Individual research project
