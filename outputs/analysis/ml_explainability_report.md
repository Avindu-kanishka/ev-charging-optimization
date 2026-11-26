# Explainable Machine Learning Research Report

## 1. Research objective

This experiment investigates which geographic and road-network characteristics contribute to predicting OSM-based urban-activity demand proxy presence in Colombo.

The target represents mapped urban-activity proxy locations, not observed EV charging sessions or electricity consumption.

## 2. Permutation feature importance

Permutation importance was evaluated using ROC-AUC on five held-out spatial folds. Higher positive values indicate a greater reduction in predictive performance when a feature is shuffled.

### 2.1 Extra Trees

| Feature | Mean importance | Across-fold SD |
|---|---:|---:|
| latitude | 0.04665 | 0.01001 |
| longitude | 0.03192 | 0.01095 |
| road_trunk_count | 0.02731 | 0.00532 |
| road_residential_count | 0.01443 | 0.00527 |
| incident_road_length_mean_m | 0.01062 | 0.00497 |
| road_unclassified_count | 0.00508 | 0.00763 |
| local_node_count_250m | 0.00425 | 0.00564 |
| road_primary_count | 0.00327 | 0.00513 |
| local_node_count_500m | 0.00265 | 0.00635 |
| incident_road_length_sum_m | 0.00218 | 0.00602 |

### 2.2 Random Forest

| Feature | Mean importance | Across-fold SD |
|---|---:|---:|
| latitude | 0.06177 | 0.00897 |
| longitude | 0.05111 | 0.01706 |
| road_trunk_count | 0.02936 | 0.00504 |
| incident_road_length_mean_m | 0.02313 | 0.00854 |
| incident_road_length_max_m | 0.01499 | 0.01037 |
| road_residential_count | 0.01019 | 0.00439 |
| incident_road_length_sum_m | 0.00920 | 0.00879 |
| road_primary_count | 0.00644 | 0.00378 |
| local_node_count_500m | 0.00471 | 0.00537 |
| local_node_count_250m | 0.00409 | 0.00318 |

## 3. Feature importance stability

### Extra Trees

| Feature | Folds with positive importance |
|---|---:|
| incident_road_length_mean_m | 5/5 |
| road_residential_count | 5/5 |
| road_trunk_count | 5/5 |
| longitude | 5/5 |
| latitude | 5/5 |
| road_unclassified_count | 4/5 |
| road_motorway_count | 4/5 |
| road_primary_count | 4/5 |
| local_node_count_250m | 4/5 |
| road_living_street_count | 4/5 |

### Random Forest

| Feature | Folds with positive importance |
|---|---:|
| incident_road_length_mean_m | 5/5 |
| road_primary_count | 5/5 |
| longitude | 5/5 |
| local_node_count_250m | 5/5 |
| latitude | 5/5 |
| road_residential_count | 5/5 |
| road_trunk_count | 5/5 |
| local_node_count_500m | 4/5 |
| incident_road_length_max_m | 4/5 |
| incident_road_length_sum_m | 4/5 |

## 4. Geographic feature ablation

The Extra Trees classifier was evaluated using the same five spatial folds with and without latitude and longitude.

| Feature configuration | Predictors | Mean ROC-AUC | SD |
|---|---:|---:|---:|
| Full Features | 21 | 0.7270 | 0.0161 |
| Road Network Only | 19 | 0.6768 | 0.0116 |

Removing geographic coordinates reduced mean ROC-AUC by 0.0502.

## 5. Research interpretation

- Geographic coordinates are highly influential predictors in both tree-based models.
- Road trunk and residential road characteristics also contribute predictive information.
- Road-network-only features retain predictive ability, although performance is lower than the full feature model.
- Local road-node density contributes less than geographic position and selected road hierarchy features in these experiments.

## 6. Implications for infrastructure planning

The results suggest that spatial context and road-network characteristics can jointly support preliminary screening of potential urban demand areas. Road-network-only predictions may also be useful where geographic coordinates are unavailable or where transferable structural patterns are being investigated.

## 7. Limitations

1. The target is an OSM-based urban-activity proxy, not measured EV charging demand.
2. Permutation importance describes predictive dependence, not causal influence.
3. Correlated predictors can share or mask importance.
4. Spatial validation within Colombo does not establish generalization to other cities.
5. External validation using actual EV charging-session data is required before operational demand-planning claims.

## 8. Conclusion

The explainability experiments show that geographic position is a major predictor of OSM-based demand proxy presence, while road-network characteristics provide additional predictive information. These findings support further investigation using real EV charging data and spatial graph-learning methods.