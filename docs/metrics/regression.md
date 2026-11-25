# Dense regression metrics

Import `mean_absolute_error`, `median_absolute_error`, `mean_squared_error`,
`root_mean_squared_error`, `max_error`, `r2_score`, and
`explained_variance_score` from `moxie.metrics`.

Targets are NuMojo vectors `(n_samples,)` or matrices
`(n_samples, n_outputs)`. A vector and a one-column matrix describe the
same output. The inputs can have different dtypes among Float32, Float64,
and integral types. Target values must be finite. Metrics read strided
views in logical row and column order.

Scalar functions return `Float64` and average outputs uniformly by
default. For custom multioutput aggregation, pass `output_weight` as a
rank-1 Float64 NDArray with one weight per output. `r2_score` and
`explained_variance_score` also accept `multioutput="variance_weighted"`.
They average uniformly if all target variances are zero. A custom
`output_weight` and variance weighting are mutually exclusive. The six
aggregating metrics have `*_raw` entry points for scikit-learn's
`multioutput="raw_values"`. These return rank-1 Float64 NDArrays in input
column order. `max_error` accepts only a single output.

The six aggregating metrics accept `sample_weight` as a rank-1 Float64
NDArray with one finite weight per sample. All-zero sample weights raise.
An average with zero total weight also raises. Weighted median uses the
averaged inverse CDF and skips zero-weight residuals when choosing the
second value at an exact split. Signed weights can yield a nonmonotone CDF.
The implementation uses the same binary search as the reference, with ties
in residual value kept in input order.

R² and explained variance default to `force_finite=True`. For a
zero-variance output this gives 1 for a zero numerator and 0 otherwise.
With `force_finite=False`, IEEE division can produce NaN or infinity.
R² with one sample emits `UndefinedMetricWarning` on stderr and returns
NaN. Its raw form returns one NaN per output even though scikit-learn
returns a scalar NaN in that case.

Calculations widen values to Float64 before subtracting. This can differ
from scikit-learn's Float32 steps and avoids integral subtraction overflow
in median and maximum error. Integers beyond Float64's exact
representation may lose precision on conversion. NumPy's reduction order
can also produce different rounding.

## Model selection

A `TypedScorer` can pass fold-local sample weights to a metric and return
its scalar result directly. Use a negative sign for a loss so
model_selection can maximize the score.

```mojo
return -root_mean_squared_error(y, estimator.predict(X), sample_weight=metadata)
```

The [weighted scorer example](../../tests/metrics/regression_scorer.mojo)
uses R² and negative RMSE with separate fit and score metadata.

## Implementation and validation

Numax's `average(xs, weights)` takes matching row-major tensors, performs
an elementwise product, and has no per-axis weighted overload.
`sum_axis` takes row-major TileTensor storage and expects the values to
reduce as input. A direct composition would require materializing
residuals, squared errors or weighted products from the original strided
inputs. The native loops perform these operations as they read the
targets. R² and explained variance use two passes to compute centered
sums without a temporary matrix.

Numax's `median` uses a host copy and has no weighted overload. NuMojo's
`median_1d` constructs a sorted array. The preprocessing
`weighted_percentile_column` helper needs a materialized residual matrix
column and separate ordering, sorted weight and cumulative weight buffers.
The metric sorts one residual list per output in place with Mojo's
standard sort. The weighted list keeps both the weight and cumulative sum.
The scalar reductions other than median require constant auxiliary storage.

Use the locked Mojo 1.0.0 / MAX 26.5.0 toolchain and set `SKLEARN_PYTHON`
to the Python built from the pinned local scikit-learn checkout.

```sh
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mkdir -p builds/metrics
mojo build -I builds/deps tests/metrics/regression.mojo -o builds/metrics/regression
mojo build -I builds/deps tests/metrics/regression_scorer.mojo -o builds/metrics/regression_scorer
"$SKLEARN_PYTHON" tests/metrics/parity.py --regression builds/metrics/regression --regression-scorer builds/metrics/regression_scorer
```

The comparison checks the reference source bytes, metrics, error cases,
warning text, raw output shapes and weighted cross-validation scores.
The Float64 comparisons use absolute and relative tolerances of
`1e-14` on bounded values, with `2e-6` for the Float32 and mixed-dtype
cases to account for widening.
