# Agglomerative clustering

`moxie.cluster` exports `AgglomerativeClustering`, `FeatureAgglomeration`,
`linkage_tree`, and `ward_tree`. The implementation is native Mojo and consumes
current Moxie and neighbor metric sources. The behavioral reference is
scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Samples and features

```mojo
import numojo as nm
from moxie.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree

var x = nm.zeros[DType.float64](nm.Shape(4, 2))
x[nm.Item(2, 0)] = 8
x[nm.Item(3, 0)] = 9
var model = AgglomerativeClustering(2, linkage="ward", compute_distances=True)
var labels = model.fit_predict(x)
var tree = ward_tree(x, return_distance=True)
var features = FeatureAgglomeration(1)
var pooled = features.fit_transform(x)
var reconstructed = features.inverse_transform(pooled)
```

Estimator dtype is a compile-time parameter, defaulting to Float64; use
`AgglomerativeClustering[DType.float32]` for Float32 inputs. Inputs are finite
rank-2 `numojo.NDArray` matrices. Sample clustering requires at least two rows;
feature clustering requires at least two columns and permits a single row.
Inputs with valid strides and offsets are supported. Non-contiguous inputs are
materialized with `NDArray.contiguous()` before dtype conversion, costing one
additional input-sized allocation. Tree helpers additionally
accept rank-1 input as a column of samples. Trees and calculations promote to
Float64, as in the reference's unstructured linkage path.

All four linkage methods are supported: `ward`, `complete`, `average`, and
`single`. Metrics are `euclidean`/`l2`, `manhattan`/`l1`/`cityblock`, `cosine`,
`chebyshev`, and dense `precomputed`. Ward requires Euclidean distance. Cosine
rejects zero vectors and norms that underflow or overflow Float64. Precomputed matrices must be square; the upper triangle
supplies distances, matching the reference. Estimator inputs must be finite
everywhere; the standalone precomputed helper ignores nonfinite entries outside
the upper triangle. Other metrics raise a clear error.

Constructor defaults match sklearn: `n_clusters=2`, `metric="euclidean"`,
`memory=None`, `connectivity=None`, `compute_full_tree="auto"`, `linkage="ward"`,
`distance_threshold=None`, and `compute_distances=False`. FeatureAgglomeration
adds `pooling_func="mean"`, the native spelling of `np.mean`. Only default mean
pooling is currently supported; other pooling functions raise. Python metric
and pooling callables are not accepted by these statically typed signatures.
`FullTree` (from `moxie.cluster._agglomerative`) represents `"auto"`, `True`, or
`False` and supports implicit conversion in the constructor. Use
`FullTree(False)` when setting an optional keyword parameter.

Exactly one of `n_clusters` and `distance_threshold` must be set. Merges at the
threshold remain separated. Threshold clustering requires a full tree and
stores distances automatically. With `connectivity=None` the whole tree is
always computed, even when `compute_full_tree=False`, as in sklearn.

`fit` and both typed and keyword `set_params` return a mutable borrow of the
estimator. `get_params` returns an independent typed record; constructor
configuration lives in `params`. Optional-valued keyword updates use nested
optionals: `set_params(n_clusters=Optional[Int](None),
distance_threshold=Optional[Float64](1))` clears the cluster count and sets a
threshold. Clone preserves configuration and clears learned state. A successful
refit replaces learned state; a validation failure preserves the previous learned tree.

Learned attributes are `labels_`, `children_`, `n_clusters_`, `n_leaves_`,
`n_connected_components_`, `n_features_in_`, and `distances_`. Labels and children
are Int64; distances are `Optional[NDArray[float64]]`, absent when not requested.
A refit without distances clears the optional. This differs from Python's
conditional attribute existence. Returned labels and parameter arrays are
independent copies. AgglomerativeClustering supplies `fit_predict`, but has no
prediction method for unseen samples. FeatureAgglomeration has no `fit_predict`.

## Feature transformation

FeatureAgglomeration fits the transposed input and assigns one label per feature.
`transform` and `fit_transform` return `(n_samples,n_clusters)` Float64 means,
including for Float32 input, matching sklearn's `bincount` mean path.
`inverse_transform` broadcasts each cluster value to its member features; it
accepts rank 1 or 2 and preserves the supplied value dtype. This reconstructs
pooled features, not the original unpooled values. With a Float32 estimator,
cast reconstructed Float64 data back to Float32 before calling `transform` again.
`get_feature_names_out` returns `featureagglomeration0`, etc.; optional input
names must match the fitted feature count. Arrays carry no input feature names.

## Tree helpers

`ward_tree(X, *, connectivity=None, n_clusters=None, return_distance=False)`
and `linkage_tree(X, connectivity=None, n_clusters=None, linkage="complete",
affinity="euclidean", return_distance=False)` return the shared `MergeTree`:

- `children`: `(n_leaves-1,2)` Int64, leaves numbered from zero.
- `n_leaves` and `n_connected_components` (one for unstructured input).
- `parents=None`, matching the unstructured reference path.
- Optional Float64 `distances`, present only when requested.

Call `.validate()` for structural checks. `linkage_tree` accepts single, average,
or complete linkage; Ward uses its separate helper. Helper `n_clusters` is
accepted but never truncates an unstructured tree. Unlike Python, no warning is
emitted for that ignored early-stop request. The record replaces Python's
variable-length return tuple.

## Scope and cost

Connectivity-constrained linkage, sparse inputs, joblib memory caching, Python
callables, metadata routing, pandas output, and `set_output` are unsupported.
Supplied dense connectivity or a memory path raises `NotImplementedError`;
there is no silent unconstrained fallback or sparse conversion.

Ward/average/complete use a nearest-neighbor chain and an O(n²) Float64 distance
matrix. Single linkage uses on-demand distances and O(n) linkage scratch beyond
the promoted input. Both use stable edge ordering and union-find relabeling;
cutting preserves sklearn's heap-based label numbering. Overflow or nonfinite
computed distances raise rather than producing an invalid tree. No performance
claim is made.

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity` from the family root.
The parity suite uses original local sklearn Python bodies and compiled Cython
MST, union-find, and descendant bodies. Only the Cython metric ABI is adapted
to SciPy `cdist`; the other linkage paths execute SciPy's implementation.
Original source revisions/hashes, tool versions, flags, seed, and case names
are retained in ignored `builds/agglomerative-parity.json`.

Children, labels, shapes, and counts compare exactly. Float64 linkage distances
and pooled means use `rtol=atol=2e-12`, allowing scalar evaluation-order rounding
on the tested bounded inputs. Tests cover ties, duplicates, identical inputs,
metric aliases, precomputed distances, both dtypes, all cluster counts selected
by the suite, thresholds, helper outputs, and feature reconstruction. Native
contracts additionally cover negative strides, invalid input, optional outputs,
clone/copy ownership, fluent generic dispatch, parameter updates, refits, and
unfitted transform errors. Both isolated consumers import and execute all four
exports. Compile-failure tests check the absence of sample prediction and
feature `fit_predict`. This is dense-subset evidence, not full sklearn parity.
