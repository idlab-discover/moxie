# KNNImputer

Import `KNNImputer` from `moxie.impute`. The implementation uses the verified
integrated `moxie.neighbors` brute-force query engine and its missing-aware Euclidean
metric. Training rows eligible to donate differ for each target feature;
neighbor distances always use the original query values. If no donor has a
defined distance, imputation uses the observed training column mean.

```mojo
var model = KNNImputer(n_neighbors=2, weights="distance")
model.fit(X)
var filled = model.transform(X_new)
```

Run `pixi run build` from the family root, then `builds/knn/knn` for a complete
example. `pixi run test` runs native contracts and isolated consumer probes;
`pixi run parity` compares against the pinned local scikit-learn source.

## Inputs and configuration

`KNNImputer[dtype=DType.float64]` accepts nonempty rank-two NuMojo NDArrays of
its selected dtype, Float32 or Float64, including strided views. Outputs have
the same dtype and row order. Integer/categorical arrays, cross-dtype calls,
sparse arrays and Python dataframe integration are not supported by this API.
Infinite input values are rejected. A finite missing marker also rejects NaN
input values, matching scikit-learn validation.

| Parameter | Default | Supported values |
| --- | --- | --- |
| `missing_values` | `MissingValue[dtype]()` (NaN) | NaN or a finite numeric marker |
| `n_neighbors` | `5` | Positive integer, capped per feature at eligible donor count |
| `weights` | `"uniform"` | `"uniform"`, `"distance"`, `"callable"` |
| `metric` | `"nan_euclidean"` | `"nan_euclidean"`, `"callable"` |
| `copy` | `True` | Both settings produce owned output and preserve input |
| `add_indicator` | `False` | Append indicators for columns missing during fit |
| `keep_empty_features` | `False` | Drop fit-empty columns, or keep and fill with zero |

`fit`, `fit_transform`, `transform`, `get_params`, `set_params`,
`get_feature_names_out`, `__sklearn_clone__` and `__sklearn_is_fitted__` are
implemented. `fit_named(X, names)` supplies native feature-name metadata.
`fit` and `fit_transform` accept an ignored optional target `y`.
`get_params()` returns the typed `KNNParams` record (`copy_input` stores `copy`);
`set_params` accepts that record or keyword updates and clears fitted state.
To clear an optional callback, use the typed record. Fitting replaces learned
state; a failed fit preserves the last successful state. Cloning copies only
configuration, whereas copying the fitted struct preserves its learned state.

Learned public attributes are `n_features_in_`, optional `feature_names_in_`,
and optional `indicator_`. Training data and missing masks are owned snapshots.
Unfitted operations and invalid configuration raise clear Mojo `Error`s.

## Native callback interface

Callbacks use the neighbors project's statically typed traits:
`KNNImputer[dtype, M, W]`, where `M` conforms to `DenseMetric` and `W` to
`NeighborWeights`. Pass `metric="callable", metric_callback=M(...)` or
`weights="callable", weights_callback=W(...)`. Default types are
`Metric[dtype]` and `BuiltinWeights`. Callback objects must be copyable.

A metric supplies `dtype`, `validate_features(n_features)` and
`distance(x, i, y, j)`. It receives full-width normalized matrices and row
indices; all configured missing markers have become NaN. Distances must be
nonnegative or NaN (undefined). This adapts Python's
row-array callback and `missing_values` keyword to the shared native trait.
A weights callback supplies `apply(distances, offsets)` and returns a rank-one
Float64 NDArray of the same length. Each call describes one receiver's selected
neighbors with offsets `[0, k]`. Distances are sorted ascending, with NaNs last.
NaN weights become zero; infinite callback weights raise an explicit error.
Unlike builtin uniform weights, custom weights may include donors whose
distance is NaN. All-NaN distances bypass weighting and use the column mean.
Callbacks depending on a Python batch shape or argpartition order require
adaptation; arbitrary Python callables are not accepted.

## Reference details and numerical differences

The reference is scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Native equal-distance boundaries
select the earliest eligible fitted row; scikit-learn's argpartition does not
promise this tie order. Tests separately assert the native deterministic rule.

The shared metric sums squared differences over jointly observed coordinates.
Scikit-learn uses a norm expansion that can lose Float32 precision for almost
equal rows. Differential tests require Float64 agreement within `3e-12` and
Float32 agreement within `3e-6` (absolute and relative) against the reference
run on the same Float32 inputs promoted to Float64. They also record differences
from the reference's native Float32 result, which can exceed ordinary rounding
tolerance because distance weighting amplifies cancellation. Identical neighbor
selection at numerical ties is not promised. The suite checks exact output
shapes and feature names against the original-dtype reference.

At the pinned revision, `get_feature_names_out` excludes fit-empty features
even when `keep_empty_features=True` retains zero-valued output columns. This
reference behavior is preserved and explicitly tested; in that case the name
count can differ from the output width.

The implementation bounds receiver batches to 128 and reuses neighbor search,
metric and builtin weighting code. It makes no benchmark-based performance
claim. IterativeImputer remains a separate deferred implementation.
