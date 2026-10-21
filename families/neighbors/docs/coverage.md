# Neighbors API coverage

Reference: local `../scikit-learn`, revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` (matching `docs/interfaces.md`).
Compare `sklearn/neighbors/__init__.py::__all__` with the explicit exports in
`DEPENDENCIES.json` and their algorithm source files. Historical raising
declarations in `_api.mojo` and `_trees_api.mojo` do not count as implementations.
The inventory is source-level coverage, not a behavioral parity percentage.
NearestCentroid additionally has native contract tests and 432 pinned-source
parity configurations; see [the centroid implementation guide](centroid.md).

The reference has 18 public names: twelve have dense implementations with
documented limits, and six are deferred under the family's
documented sparse scope. Export counts are not behavioral parity percentages.

| Reference public name | Coverage | Implemented operations or missing work |
| --- | --- | --- |
| NearestNeighbors | Partial dense | fit, kneighbors, radius_neighbors; brute/KDTree/BallTree backends; no graph methods |
| KNeighborsClassifier | Partial dense | fit, predict, predict_proba, score and queries; string/Int64/floating labels |
| KNeighborsRegressor | Partial dense | fit, predict, score and queries |
| RadiusNeighborsClassifier | Partial dense | fit, predict, predict_proba, score and queries; string/Int64/floating labels |
| RadiusNeighborsRegressor | Partial dense | fit, predict, score and queries |
| LocalOutlierFactor | Partial dense | fit, fit_predict, kneighbors, novelty-gated predict/decision_function/score_samples; no kneighbors_graph |
| KernelDensity | Partial dense | fit, score_samples, score, seeded/default-stream Gaussian/tophat sample; six Euclidean kernels |
| NearestCentroid | Partial dense | fit, predict, score, both metrics, shrinkage, priors and Euclidean decision/probability methods; string/Int64/floating labels |
| NeighborhoodComponentsAnalysis | Implemented dense slice | All initialization modes, supervised fit/transform, warm starts, native callbacks; [numerical/API limits](nca.md) |
| KDTree | Partial dense | Construction, metrics, single/dual queries, radius/counts, density, correlation, inspection and counters; see [KDTree](kdtree.md) |
| BallTree | Partial dense | Construction, queries, density and inspection; see [BallTree](balltree.md) |
| VALID_METRICS | Partial dense | Public backend-specific metric registry; see [BallTree](balltree.md) |
| VALID_METRICS_SPARSE | Deferred | Sparse metric registry |
| KNeighborsTransformer | Deferred | Sparse graph transformer |
| RadiusNeighborsTransformer | Deferred | Sparse graph transformer |
| kneighbors_graph | Deferred | Sparse graph function |
| radius_neighbors_graph | Deferred | Sparse graph function |
| sort_graph_by_row_values | Deferred | Sparse graph utility |

Existing estimators also expose typed get_params/set_params (record and keyword forms) and unfitted cloning.
NearestNeighbors, LOF and KNN/radius estimators accept native callable metrics
with brute, auto and BallTree selection; KDTree rejects them. General classification
labels use the shared label owner. Self queries infer omitted standardized-Euclidean
V and Mahalanobis VI. KDE supports both trees and default-stream sampling.
Concrete capabilities, stored callable weights and remaining native adaptations
are documented in the API guide. Python reflection and metadata routing remain
outside the interface; export coverage does not establish exhaustive edge parity.
See [KDE sampling](kde-sampling.md) for the 216 sampling parity configurations. See
[estimators.md](estimators.md) for precise type, shape, parameter and tolerance
adaptations and [scope.md](scope.md) for dense versus deferred work.

## NearestCentroid implementation contract

Implemented at `algorithms/centroid/src/moxie/neighbors/_nearest_centroid.mojo`
and exported through the family assembly. It is independent of spatial-tree
acceleration and does not require NCA's iterative optimization machinery.

The reference constructor is
`NearestCentroid(metric="euclidean", *, shrink_threshold=None, priors="uniform")`.
The source default is uniform despite contradictory wording in its priors
parameter description. Euclidean centroids are class means; Manhattan centroids
are feature-wise medians. Preserve sorted classes, Float64 centroids,
within_class_std_dev_, deviations_, class_prior_ and n_features_in_.

Uniform-prior prediction uses nearest unnormalized centroids. Nonuniform-prior
prediction uses squared distances after within-class standard-deviation scaling,
with a log-prior correction. These paths must be tested separately: implementing
only nearest-mean prediction would miss the current reference contract.
decision_function, predict_proba and predict_log_proba are available only for
the Euclidean metric. Binary and multiclass decision shapes differ.

Reference sources are `_nearest_centroid.py`, its inherited
`DiscriminantAnalysisPredictionMixin` in `sklearn/discriminant_analysis.py`,
`sklearn/neighbors/tests/test_nearest_centroid.py`, and the nearest-centroid user
guide. References and numerical choices are retained locally under
`docs/private/centroid/`. Contract and parity coverage includes
both metrics, shrinkage, uniform/empirical/explicit priors, class ordering,
probabilities, invalid/unfitted inputs, constant features, zero pooled variance,
singleton classes, refits, parameter changes, cloning and strided inputs.

NeighborhoodComponentsAnalysis provides a dense supervised embedding with all
initialization modes, warm starts, native callbacks, and feature names. See
[NCA API and numerical limits](nca.md).
