# Dense neighbors scope — PARTIAL

Implemented public slices are NearestNeighbors, KNeighborsClassifier,
KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor,
LocalOutlierFactor, KernelDensity, NearestCentroid, NeighborhoodComponentsAnalysis, KDTree, BallTree and VALID_METRICS. See [the API guide](estimators.md) for
supported methods, type adaptations, parameters and numerical limitations.

Nearest/radius queries and supervised prediction support Float32/Float64 dense
features, precomputed distances, rank-one/two numeric targets, ordered classes,
uniform/distance/native callable weighting, native callable distance metrics on
brute, auto and BallTree queries, LOF and supervised estimators, self exclusion, duplicates and empty
neighborhoods. LOF provides training factors, contamination thresholds and gated
novelty inference. KDE provides six metric kernels (Euclidean probability normalization), weighted fitting, numeric
and rule-based bandwidth, summed log likelihood and seeded and default-stream Gaussian/tophat sampling.
All engines are pure Mojo on accepted NuMojo arrays. KDTree and BallTree support
explicit selection and eligible auto queries; see [KDTree](kdtree.md) and
[BallTree / VALID_METRICS](balltree.md).

Dense support includes string/Int64 targets independent of feature dtype,
multilabel target dispatch, native callable metrics and stored callable weights,
self-query standardized-Euclidean variance inference, arbitrary positive Minkowski
powers through native Float64 power, and default-stream KDE sampling. Capability
traits and native API adaptations are documented in [estimators.md](estimators.md).
This does not claim exhaustive sklearn parity: Python reflection/metadata routing,
warning emission and the documented numerical-domain restrictions remain outside
the supported native interface. `n_jobs` remains a serial scheduling hint.

Deferred by the retained dense-only contract: sparse
feature/precomputed storage, VALID_METRICS_SPARSE, KNeighborsTransformer,
RadiusNeighborsTransformer, kneighbors_graph, radius_neighbors_graph,
sort_graph_by_row_values, and graph-returning estimator methods. No implicit
densification or alternate dense graph output is provided. Historical API
inventory declarations are not assembled as implemented exports.

Validation commands are `pixi run build`, `pixi run test`, `pixi run parity`,
`pixi run start`, and the query-engine's `pixi run test` / `pixi run parity`.
Build/test compose two isolated downstream consumers with real estimator calls,
verify shared ownership, reject drift/conflicting exports, compile all owned
modules and reject sparse imports. Parity covers 192 supervised configurations across auto, KDTree and BallTree,
LOF reachability/novelty calculations, 576 KDE configurations across three metrics
and backend selections, and 432 centroid configurations. KDTree adds 336 independent configurations and 205 original local sklearn
test invocations with a compiled pinned oracle. BallTree adds 602 independent
configurations and 238 original test invocations. Query-engine
parity compares 108 dtype/metric configurations. Reference tests use pinned
sklearn generic method bodies and independent SciPy formulas; they do not exercise
all compiled sklearn paths or establish full estimator-check compatibility.
No speed or memory benchmark claim is made. Broader extreme/rank/stride coverage
and performance measurement remain unfinished.
