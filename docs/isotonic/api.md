# Isotonic regression

`moxie.isotonic` implements `IsotonicRegression`, `isotonic_regression`, and
`check_increasing`. Behavior is based on scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The current
[shared estimator contract](../interfaces.md) defines fluent mutation methods.

## Data and configuration

Inputs and outputs use `List[Float64]` by default.
`IsotonicRegression[DType.float32]` uses `List[Float32]` throughout, including
learned thresholds and predictions. Both helpers infer the list dtype.
Estimator X also accepts `List[List[Scalar[dtype]]]` with exactly one element
per row, flattened internally. Predictions and transformations always return
one-dimensional lists. Integer or mixed-dtype inputs and other containers require explicit
conversion by the caller to the chosen Float32 or Float64 list type.
This retains the family List interface and PAVA workspace; no generic storage
or NuMojo operation is replaced. Float32 PAVA widens to the existing Float64
solver, then rounds before clipping, matching the SciPy-backed reference.
Tie accumulation and interpolation retain input precision; score accumulates
in Float64 and can differ from reference Float32 reductions in low bits.
Inputs must be finite and nonempty; X, y, and weights must have matching lengths.
No input list is mutated or retained by reference.

```mojo
var model = IsotonicRegression(
    y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip"
)
model.fit(X, y, sample_weight=weights)
var predictions = model.predict(query)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `y_min`, `y_max` | `None` | Optional Float64 lower/upper bounds on fitted targets |
| `increasing` | `True` | Bool or `"auto"`; automatic direction is the sign of Spearman's rank correlation |
| `out_of_bounds` | `"nan"` | `"nan"`, `"clip"`, or `"raise"` for prediction outside the training domain |

Bounds apply after the isotonic fit. Infinite bounds are allowed; NaN bounds
are rejected. As in NumPy clipping, `y_min > y_max` produces the upper bound.
Use floating-point literals for optional bounds (`0.0`, not `0`).

## Estimator methods and state

`IsotonicRegression` conforms to `IsotonicRegressorAPI` and
`WeightedSupervisedEstimator`. The family trait still requires only
`RegressorMixin`, `TransformerMixin`, and `FitTransformer`.

- `fit(X, y)` and `fit(X, y, sample_weight=weights)` mutate in place and return a mutable borrow of the estimator. Both `set_params` overloads do the same, supporting chained calls. Positive
  weights contribute; zero and negative weights are removed, matching the
  reference estimator. At least one positive weight must remain.
- `predict(X)` and `transform(X)` return the same piecewise linear values in
  input order. `fit_transform(X, y)` also accepts the weighted-fit overload.
  Omitting y raises an error.
- `score(X, y, sample_weight=None)` computes weighted R². Constant targets
  produce 1 for perfect predictions and 0 otherwise; fewer than two samples
  produce NaN. Nonfinite predictions or weights with zero total are rejected.
- `get_params(deep=True)` returns an independently owned
  `IsotonicRegressionParams` record. `set_params(params)` replaces configuration;
  parameter changes do not themselves refit or clear learned state.
- Named `set_params` updates support supplied non-None values. To clear a bound,
  set `params.y_min = None` or `params.y_max = None` on the typed record and pass
  it back. For a named direction update use `Increasing(False)` or
  `Increasing(String("auto"))`, importing `Increasing` from `moxie.isotonic`.
- `clone(model)` returns an unfitted model with copied configuration;
  `model.copy()` copies fitted state into independent buffers.
- `get_feature_names_out()` returns `["isotonicregression0"]` after fitting.

A successful fit replaces `X_min_`, `X_max_`, `X_thresholds_`, `y_thresholds_`,
and `increasing_`. Failed fits preserve the previous valid state. Learned fields
exist before fitting because Mojo structs have fixed fields; they are valid
only when `__sklearn_is_fitted__()` is true. Inference checks fitted state.

`f_(T)` is a method evaluating the fitted interpolator directly. It does not
apply input clipping. It is not a stored Python callable. It accepts the same
finite, nonempty list contract as prediction.

## Ordering, ties, and boundaries

Fit sorts observations by X, then by y. Duplicate X values are replaced by
their weighted mean targets and total weights, using the reference absolute
resolution tolerance of `1e-15` (Float64) or `1e-6` (Float32), measured
from each group's first X. A stack-based
PAVA solver pools adjacent violating blocks to their weighted mean. Interior
points of constant fitted plateaus are removed while retaining both endpoints
needed for interpolation.

For `increasing="auto"`, average ranks handle ties. Direction is determined
from all original samples, before weight filtering. Zero rank correlation
selects increasing; an undefined correlation (constant input or singleton)
selects decreasing.

Prediction interpolates between retained thresholds, uses exact values at
thresholds, and handles out-of-range inputs according to `out_of_bounds`.
A single retained X threshold predicts a constant everywhere for all three
modes, matching the reference. The training domain excludes discarded weights
and uses the representative X values after duplicate consolidation.

The reference binds interpolation's `bounds_error` at fit time. This port
preserves that detail: switching `out_of_bounds` to `"clip"` takes effect
immediately, while switching the interpolator's raise behavior requires refit.

The solver uses O(n) time and storage; sorting and automatic ranking use
O(n log n), and prediction uses binary search in the retained thresholds.
These are algorithmic bounds, not measured performance claims.

## Public helpers

`isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
increasing=True)` fits an **already ordered** target sequence. All options
are keyword-only. Unlike the estimator, this helper requires strictly positive
weights, matching the reference's SciPy implementation. It returns a new
same-dtype list of the same length and never sorts its inputs.

`check_increasing(x, y)` returns a Bool from Spearman's rank correlation.
As a deliberate validation difference, it rejects empty and nonfinite inputs.
Python's confidence-interval and constant-input warnings are not emitted;
small-sample R² warnings are also not emitted. The corresponding numerical
results are covered by native and reference-comparison tests.

## Scope and validation

Python bindings, automatic heterogeneous dtype coercion, scalar sample
weights, pandas output, metadata routing and serialization are not implemented.
Pass weights as a same-dtype list. `get_feature_names_out(input_features)`
accepts and ignores the optional names, as the reference does. Sparse inputs
are unavailable in this typed API; see [dense scope](scope.md). Errors use
Mojo `Error` with `ValueError`/`NotFittedError` prefixes. See the shared guide
for in-place fitting and typed configuration differences.

`pixi run test` covers shared contracts, trait conformance of the concrete
estimator, fixed numerical examples, validation, clone/copy ownership, repeat
fits, 1d/one-column equivalence, direction, bounds, and weighted R².
`pixi run parity` generates 623 Float64 and 456 Float32 deterministic cases against the exact reference
Python module, compiling the pinned local Cython tie engine and using the scikit-learn support wheel and SciPy from
Moxie's locked Pixi environment. Float64 comparisons use `atol=rtol=2e-12`:
PAVA block-reduction order can differ from SciPy, so roundoff is allowed;
Float32 uses `atol=rtol=2e-6` for rounding and reduction-order differences.
Threshold counts, X thresholds, domain, direction, and NaN locations must match. Fixture generation
and subprocess orchestration are test tooling only.

Reference sources: `../../../scikit-learn/sklearn/isotonic.py`,
`../../../scikit-learn/sklearn/_isotonic.pyx`,
`../../../scikit-learn/sklearn/tests/test_isotonic.py`, and the isotonic user guide.
The weighted least-squares PAVA formulation is also documented by
[SciPy](https://scipy.github.io/devdocs/reference/generated/scipy.optimize.isotonic_regression.html).
Local reference copies, mathematical citations, revisions, and licenses live
under ignored `docs/private/isotonic/`.

To reproduce the small native timing and memory baseline after parity:

```sh
pixi run python tools/isotonic_benchmark.py --processes 1 --values 3 --warmups 1 --loops 3 -o benchmarks/isotonic/baseline.json
```

The driver excludes process startup from timings and records whole-process peak
RSS separately. These short runs do not establish performance superiority.
