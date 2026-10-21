# Dense neighbor estimators

The assembled `moxie.neighbors` namespace exports `NearestNeighbors`,
`KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighborsClassifier`,
`RadiusNeighborsRegressor`, `LocalOutlierFactor`, `KernelDensity` and
`NearestCentroid`, `KDTree`, `BallTree` and `VALID_METRICS`.
These are pure Mojo implementations, separate from the historical raising API
inventory in `_api.mojo`. Build only through `pixi run build`; sources remain
in their algorithm owners and are composed through explicit export manifests.

## Queries and supervised prediction

Use `Estimator[DType.float64]` (the default) or `[DType.float32]` with NuMojo
arrays. Feature matrices are rank two; target arrays are rank one or rank two.
Classification uses the shared `moxie._labels` domains. KNN/radius classifiers
accept a third type parameter `Y` (default `NDArray[dtype]`): choose
`NDArray[DType.int64]`, either floating dtype, or `StringTarget`. For example,
`KNeighborsClassifier[DType.float32, Metric[DType.float32], StringTarget](3)`
fits Float32 features with string labels. Int64 labels never pass through Float64.
Class order is ascending independently for each output; strings use UTF-8
lexicographic order. Predictions preserve target storage and dtype. Probabilities
are Float64. Regression accumulates in Float64.

`fit(X, y)` snapshots inputs; failed validation preserves the preceding fit.
`clone(model)` returns an unfitted independent configuration. `get_params()` returns
a typed record; supervised records contain `neighbors: NeighborParams` and
`weights`. Modify that record and call `set_params(params)`, then refit when changing
fit-time options. Parameter updates preserve learned state, as in sklearn.
Fit and set_params return a mutable borrow of self, supporting fluent chaining.
Read constructor options through `get_params()`; direct
Python-style attribute assignment and partial keyword `set_params` are not provided.

`kneighbors(X=None, n_neighbors=None, return_distance=True)` returns `TopKResult`.
`radius_neighbors(X=None, radius=None, return_distance=True, sort_results=False)`
returns `RadiusResult` with ragged offsets. Concrete return records are an explicit
adaptation. KNN and radius estimators implement their corresponding query,
classifier/regressor and probability traits. NearestNeighbors implements
UnsupervisedEstimator and KNeighborsQuery. Its radius method returns the concrete
RadiusResult; the inherited query traits cannot express two different result
records on one type. LOF implements NoveltyDetector (with novelty gating), and
KDE implements NeighborDensity. `None`
excludes the row's own training index while retaining duplicates. Explicit arrays
include self. Equal distances use ascending training index; sklearn backend tie
selection is unspecified. Radius membership is inclusive. Negative/NaN radii are
rejected; infinity is accepted. Result distances retain feature dtype.

`algorithm="auto"` selects KDTree, then BallTree for eligible low-dimensional
inputs; other cases use streaming brute force. Explicit `"brute"`, `"kd_tree"`
and `"ball_tree"` are supported. See [KDTree](kdtree.md) and
[BallTree](balltree.md) for selection, metrics and tree contracts.
`n_jobs` remains a scheduling hint; execution is serial. `leaf_size` controls
tree partitioning and has no brute-force effect.

Metrics use the shared engine. The default Minkowski p=2 resolves to Euclidean.
Dense `metric="precomputed"` accepts a square training distance matrix and
queries with one column per training row. `metric_params` is an optional typed
`Metric[dtype]` record containing the full effective metric configuration; when
supplied, it replaces `metric`/`p` rather than acting as a Python keyword dict.
Explicit V/VI and feature weights are supported by that record. See the engine's
[scope](../shared/query_engine/docs/scope.md) for remaining metric limitations.
Native callable distances are supported by `NearestNeighbors[dtype, M]`,
`KNeighborsClassifier[dtype, M]`, `KNeighborsRegressor[dtype, M]`,
`RadiusNeighborsClassifier[dtype, M]`, `RadiusNeighborsRegressor[dtype, M]`,
`LocalOutlierFactor[dtype, M]` and `BallTree[dtype, M]`.
The second type parameter defaults to `Metric[dtype]`, preserving built-in calls.
For a custom metric, pass its configuration as `metric_params`. `auto`, `brute`
and `ball_tree` are supported; KDTree rejects callables. Auto uses the same
sample-count/dimension heuristic as built-in metrics. Callable BallTree traverses
without triangle pruning, so callbacks need not obey the triangle inequality.
`metric_params` supplies the complete distance configuration, replacing `metric`
and `p`. The custom metric type is fixed at compile time; parameter updates can
change its state but cannot change its type.

Implement the existing `moxie.neighbors._contracts.DenseMetric` trait:

- `comptime dtype` must match the estimator's Float32 or Float64 dtype.
- `validate_features(n_features: Int) raises` validates configuration on fit.
- `distance(x: NDArray[dtype], x_row: Int, y: NDArray[dtype], y_row: Int)
  raises -> Float64` returns the actual distance between the indicated rows.
- `copy()` must independently own any mutable configuration. Callbacks must not
  mutate input arrays or configuration during queries. Honor array strides and
  offsets; inputs need not be contiguous.

This is a native struct interface, not a Python callable or keyword dictionary.
No default constructor, metric-name field, symmetry, triangle inequality or
zero self-distance is required. Query evaluation uses `(query, training)` order;
self queries exclude the matching training index before evaluating distances.
Feature values must be finite. Negative or NaN callback distances raise; positive
infinity is allowed. Callback errors propagate. These output restrictions are
explicit native adaptations, not claims that sklearn validates arbitrary callable
returns the same way.

The estimator snapshots metric configuration on construction and fit. `get_params`,
`set_params` and unfitted cloning copy configuration; changing parameters leaves
the old fitted metric active until a successful refit. `effective_metric_` is
`"callable"`, and `effective_metric_params_` is an owned copy of `M`. Mutating this
inspection attribute does not change the fitted engine. Missing custom
`metric_params` raises on construction. See
[WeightedL1](../examples/callable_metric.mojo) for a complete stateful metric:

```mojo
import numojo as nm
from callable_metric import WeightedL1
from moxie.neighbors import NearestNeighbors

var weights = nm.ones[DType.float64](nm.Shape(2))
weights.unsafe_set(0, 3)
var model = NearestNeighbors[DType.float64, WeightedL1[DType.float64]](
    n_neighbors=2, algorithm="brute",
    metric_params=WeightedL1[DType.float64](weights.copy()),
)
model.fit(X)
var neighbors = model.kneighbors(Q.view())
```

The example module is in `examples/`; include that directory when compiling your
example alongside the assembled `builds/deps` namespace. Native contracts run in
`pixi run test`, including both isolated consumers. `tests/parity_callable_metrics.py`
compares both dtypes, KNN/radius queries, self-exclusion, classifier probabilities,
regression, uniform/distance/native callable weights, multi-output targets and cloned/refitted
models with the compiled pinned sklearn checkout. Run it after `pixi run build`
with that oracle on `PYTHONPATH` (or via `SKLEARN_ORACLE_PYTHON` and `pixi run parity`).

`predict(X=None)` supports uniform and inverse-distance weights. When a query
has exact matches, distance weighting assigns weight only to those matches.
For a native callable implement `NeighborWeights.apply(flat_float64_distances,
offsets) -> NDArray[float64]`, then call `predict_with_weights(X, weighting)` or
`predict_proba_with_weights(X, weighting)`. To store configuration, add the last
type parameter `W` and pass `weighting=...` to the constructor. For classifiers
this follows `Y`; for regressors it follows `M`. `get_params`, `set_params` and
clone preserve this owned configuration; ordinary predict/probability/score calls
use it. An absent `weighting` uses the built-in `weights` string. Weights must be finite,
nonnegative and shape preserving. Classification rejects a nonempty neighborhood
whose weights sum to zero. Negative/nonfinite callable or score weights are an
explicit unsupported extension of sklearn's more permissive weighting surface.

`classes_` contains one target-domain vector per output (StringTarget for strings);
`predict_proba` returns a list of Float64 arrays, one per target
output, including single-output classification. Regression preserves target rank; classification flattens a single-column
target like sklearn. Classification `score` is accuracy (subset accuracy for
multilabel indicators); multiclass-multioutput scoring raises like sklearn,
although prediction supports those targets. Regression uses uniformly averaged R2, with sklearn's finite
constant-target convention and NaN for fewer than two samples. Both accept
optional sample weights. Shared target-type dispatch recognizes numeric multilabel
indicators; multiclass-multioutput string and numeric targets remain predictable
but cannot be scored by accuracy, matching sklearn.

Radius regression returns NaN for empty neighborhoods. Radius classification
raises by default; pass `outlier_label` as a one-dimensional array with one label
per output, or `outlier_most_frequent=True` for the reference's `"most_frequent"`
setting. An outlier label outside the fitted classes yields all-zero probabilities
and the requested prediction. Native warnings are not emitted.

## LocalOutlierFactor

LOF uses the same exact queries. `n_neighbors_` is clipped to n_samples-1;
singleton fitting fails. Training factors are available as
`negative_outlier_factor_`. `contamination=None` represents sklearn's `"auto"`
and gives `offset_=-1.5`; a value in (0,0.5] uses the linear percentile.
`fit_predict` requires `novelty=False`; `score_samples`, `decision_function` and
`predict` require `novelty=True`. Novelty queries include training matches when
explicitly supplied, preserving the distinction from training factors. Scores
retain feature dtype; ±1 predictions are floating arrays, a native label-type
adaptation. Warning emission is not implemented; callable metrics use the same backend
selection and configuration rules as NearestNeighbors.

## KernelDensity

KDE performs exact metric-kernel summation with log-domain stabilization and a
KDTree or BallTree traversal stack or queue. Its worst-case work is
O(queries * training rows * features). All six kernels
are supported: Gaussian, tophat, Epanechnikov, exponential, linear and cosine.
Input may be Float32 or Float64; output log densities and samples are Float64.
`score` sums log densities. Weighted fitting requires finite nonnegative weights
with positive total; all-zero weights explicitly raise. Numeric bandwidth uses
`bandwidth`; `bandwidth_rule="scott"` or `"silverman"` represents the reference's
string bandwidth alternatives. Constructor tolerance and traversal options are
used by the tree density routine. `algorithm="auto"`, `"kd_tree"` and
`"ball_tree"` support their advertised metrics; probability normalization is
correct only for Euclidean distance. Optional typed `metric_params: Metric[dtype]`
replaces the full metric configuration. Fitted `tree_` is an optional
`Variant[KDTree[dtype], BallTree[dtype]]`;
compact-support kernels prune disjoint tree bounds when supported. Exact summation satisfies tolerances.

Gaussian and tophat `sample(n_samples, random_state=integer_seed)` and
`sample_with_rng(n_samples, rng)` use the shared MT19937 implementation, preserving
the reference's draw order. Omitting random_state advances the shared process-default
stream, and seed_global_random_state resets that stream. Sampling other kernels
is unsupported by the reference as well.

## NearestCentroid

See [NearestCentroid](centroid.md) for supported operations, priors, shrinkage,
label/dtype conventions, lifecycle semantics and validation limits.

## Validation and limits

Run `pixi run test` and `pixi run parity`. The latter uses the existing query-engine
Pixi environment for NumPy/SciPy reference testing only; install that locked
environment if absent. No Python executes in fitting, querying or prediction.

Prediction parity executes the pinned sklearn generic regression/probability
method bodies, using independent SciPy distance matrices. It covers 192 combinations
of dtype, KNN/radius, uniform/distance/callable weights, target rank, self/explicit
queries and feature/precomputed input. It does not exercise sklearn's compiled
accelerators or prove complete wrapper compatibility. LOF executes pinned
reachability and novelty-score method bodies. KDE checks the local Cython kernel
formulas and executes the pinned Python sampling method. Native tests cover
validation, cloning, refitting, snapshots, scores, strides and empty neighborhoods.

Query/prediction tolerances are rtol=atol=2e-12 for Float64 and 3e-6 for Float32,
reflecting native accumulation/output rounding. KDE, the spatial-tree kernel sums
and NearestCentroid probabilities use Numax `core.libm` exp/log (fdlibm, within one
ulp of glibc) instead of Mojo 1.0 `std.math` exp/log/pow, whose Float64 results were
measured ~1e-10 relative off (KDE log densities then differed from sklearn by up to
3e-9). Scott/Silverman bandwidths are `exp(log(n) * -1/(d+4))` in that libm. KDE
log-density parity uses rtol=atol=1e-12 in Float64. Not correctly rounded; no
performance or exhaustive extreme-value parity claim is made.

Tophat sampling contracts and validation: [KDE sampling](kde-sampling.md).

NeighborhoodComponentsAnalysis provides a dense supervised embedding with all
initialization modes, warm starts, native callbacks, and feature names. See
[NCA API and numerical limits](nca.md).
