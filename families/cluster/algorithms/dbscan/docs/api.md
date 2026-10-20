# Dense DBSCAN

Import `DBSCAN` and `dbscan` from `moxie.cluster`. Both support
`NDArray[Float32]` and `NDArray[Float64]`, with Int64 labels and core indices.
The implementation consumes the live neighboring repository's `NearestNeighbors`
radius queries, including brute-force, KDTree, BallTree and automatic selection.
Sparse data and sparse precomputed distances remain deferred.

```mojo
import numojo as nm
from moxie.cluster import DBSCAN, dbscan

var x = nm.zeros[DType.float64](nm.Shape(3, 1))
x[nm.Item(1, 0)] = 1
x[nm.Item(2, 0)] = 1
var model = DBSCAN(1, min_samples=2)
model.fit(x)
# labels_: [0, 0, 0]; core_sample_indices_: [0, 1, 2]
var result = dbscan(x, 1, min_samples=2)
```

`DBSCAN(eps=0.5, *, min_samples=5, metric="euclidean", metric_params=None,
algorithm="auto", leaf_size=30, p=None, n_jobs=None)` preserves the estimator's
constructor defaults. The helper defaults to `metric="minkowski", p=2`, like
sklearn. `p=None` resolves to 2 for the native query backend.

`fit(X, y=None, sample_weight=...)` mutates and returns a borrow of the same estimator; `y` is ignored.
Both `set_params` overloads likewise support fluent chaining.
`fit_predict` returns an independent label array. There is no unseen-data
`predict`, `transform`, or `score`. The helper returns a typed `DBSCANResult`
with `core_sample_indices` and `labels` rather than a Python tuple.
Weights are optional rank-one arrays with the input dtype, including finite
negative and zero values. Scalar broadcasting and implicit dtype conversion
are not provided. Both a neighborhood's count and its signed weight sum use
an inclusive `min_samples` threshold. Weight reduction and weighted threshold
comparison retain NumPy's input precision and pairwise order; it does not take absolute values or widen Float32.

Training neighborhoods include self when its distance is within `eps`.
Duplicate samples remain separate neighbors. Dense precomputed matrices must
be finite, nonnegative and square; supplied diagonals and asymmetry are retained,
as in sklearn's dense execution. A diagonal exceeding `eps` is excluded.
Tree algorithms reject precomputed distances. Core seeds are visited in sample
order and expanded with sklearn's depth-first stack discipline. Non-core border
points are labeled but never expanded, and keep the first cluster that reaches
them. Labels start at zero; noise is -1; core indices are ascending.

`labels_`, `core_sample_indices_`, `components_`, and `n_features_in_` are exposed.
Components are independently owned rows of the input, including complete distance
rows for precomputed input. With no cores their shape is `(0, n_features)`.
They retain the compile-time dtype even when empty (sklearn's empty-array branch
uses Float64). Before fitting, arrays are empty and `__sklearn_is_fitted__()` is
false. Successful refits replace state; validation failures retain prior state.

Constructor configuration lives in `params`, separate from learned state.
`get_params()` returns an owned `DBSCANParams[dtype]`; `set_params(record)` and
keyword updates are supported. Nullable keyword updates use nested optionals
so callers can distinguish unchanged from explicitly reset to `None`.
`clone(model)` copies configuration and returns an unfitted estimator.
Python dictionaries, reflection, metadata routing and feature names are deferred.

Supported metrics are Euclidean, Manhattan, Chebyshev, Minkowski (including
feature weights), their neighbor aliases, and dense precomputed distances.
Other metrics raise a clear `NotImplementedError`; callable metrics and inferred
metric parameters remain deferred. Typed
`metric_params` uses its native `Metric[dtype]` record, which includes the metric
name, power and optional feature weights/variance/inverse-covariance arrays;
its metric name must agree with `metric`. Its power and weights specify
the effective Minkowski configuration rather than a Python dictionary. The query backend currently executes serially; `n_jobs` is retained
and zero is rejected. Query distances inherit the neighbor family's numerical
limits; arbitrary near-roundoff radius boundaries are not a full sklearn parity
claim. Exact representable boundaries and adjacent `eps` values are tested.

The algorithm retains all radius indices (worst-case quadratic memory), a query
snapshot, and an explicit query copy required by the neighbor API's owned
optional argument. It does not densify sparse data or introduce a second search
engine. Positive/negative input strides are supported through NuMojo indexing.

Run `pixi run build`, `pixi run test`, and `pixi run parity`. To repeat just the
DBSCAN differential suite after building, run
`pixi run python algorithms/dbscan/tests/parity.py`. It executes original pinned
sklearn `DBSCAN.fit` and compiled `_dbscan_inner.pyx`, with independent SciPy
radius membership. This isolates clustering correctness; it does not execute
sklearn's complete estimator validation or its tree implementations. Reports,
source hashes and environment details are written to `builds/dbscan-parity.json`.
Native contracts additionally cover lifecycle, clone, ignored targets, weights,
strides, ownership, invalid inputs and the absence of prediction. Existing KMeans
numerical gaps remain documented in [its API](../../kmeans/docs/api.md).
