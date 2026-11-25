# Isotonic regression

`moxie.isotonic` implements `IsotonicRegression`, `isotonic_regression`, and
`check_increasing`. Behavior is based on scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The current
[shared estimator contract](../interfaces.md) defines fluent mutation methods.

## Data and configuration

The estimator traits use `numojo.NDArray[dtype]` with Float64 by default.
Use `IsotonicRegression[DType.float32]` for Float32. Pass X as `(n,)` or
`(n, 1)`, and pass y and sample weights as `(n,)`. Predictions and
transformations return a new `(n,)` array of the estimator dtype. The
estimator reads strided views in logical order. It materializes input vectors
with NuMojo's native `NDArray.tolist()` for the existing List-based solver and
uses `numojo.array` to construct results.

Concrete List overloads accept `List[Scalar[dtype]]` and, for the applicable
X arguments, one-column `List[List[Scalar[dtype]]]`. These overloads return
lists for prediction and transformation. Learned thresholds use Lists.
Convert integer or mixed-dtype inputs to the chosen Float32 or Float64 dtype.
Float32 PAVA uses the Float64 solver and rounds before clipping, matching
the SciPy-backed reference.
Tie accumulation and interpolation retain input precision; score accumulates
in Float64 and can differ from reference Float32 reductions in low bits.
Fit, predict, transform and score require finite, nonempty inputs. X, y and
weights must have matching lengths where supplied.
No input is mutated or retained by reference.

```mojo
var model = IsotonicRegression(
    y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip"
)
_ = model.fit(X, y, sample_weight=weights)
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
- `set_output(transform="default")` selects native output for `transform`
  and `fit_transform`. The output container follows the overload you call,
  with NDArray input returning NDArray and List input returning List. Omit
  `transform` or pass `None` to retain the current mode. The setter returns
  a mutable borrow for chaining. Pandas, Polars and unknown modes raise a
  Mojo `Error` at the setter.

A successful fit replaces `X_min_`, `X_max_`, `X_thresholds_`, `y_thresholds_`,
and `increasing_`. Failed fits preserve the previous valid state. Learned fields
exist before fitting because Mojo structs have fixed fields; they are valid
only when `__sklearn_is_fitted__()` is true. Inference checks fitted state.

`f_(T)` is a native method that evaluates the fitted interpolator without
input clipping. Empty queries return empty output. With multiple thresholds,
NaN queries return NaN and infinite queries follow the fitted out-of-bounds
behavior. With one threshold, all query values return the fitted constant.

### Labeled transforms

Use `transform_table(X)` or `fit_transform_table(X, y, sample_weight=weights)`
to return an owned [`ColumnTable`](../compose/named-columns.md). The methods
accept the standard dense NDArray shapes. Each result has shape `(n, 1)`,
preserves the estimator dtype and input row order, and names the column
`isotonicregression0`. The training weights are optional. The explicit table
methods give the labeled result a static Mojo type. Rows are indexed by
position.

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

After fitting, read `model.fit_warnings` for the reference warning categories
and messages in occurrence order. An automatic fit can record
`ConstantInputWarning` and a `UserWarning` about the 95% Spearman confidence
interval. A successful refit replaces the records, a failed fit preserves
them, and a fitted copy owns its own records. Unfitted clones start empty.
See the [shared fit-warning contract](../interfaces.md#fit-warnings).

Prediction interpolates between retained thresholds, uses exact values at
thresholds, and handles out-of-range inputs according to `out_of_bounds`.
A single retained X threshold predicts a constant everywhere for all three
modes, matching the reference. The training domain excludes discarded weights
and uses the representative X values after duplicate consolidation.

The interpolator binds its raise behavior during fit and when loading saved
state. Switching `out_of_bounds` to `"clip"` takes effect immediately for
`predict` and `transform`.

The solver uses O(n) time and storage; sorting and automatic ranking use
O(n log n), and prediction uses binary search in the retained thresholds.
These are algorithmic bounds, not measured performance claims.

## Public helpers

Both public helpers accept same-dtype Float32 or Float64 vectors as Lists or
one-dimensional NDArrays. Dense views are read in logical order. Pass the
same container type for every input of a helper. The estimator also accepts
one-column X, while the helpers require one-dimensional vectors.

`isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
increasing=True)` fits an **already ordered** target sequence. All options
are keyword-only. Unlike the estimator, this helper requires strictly positive
weights, matching the reference's SciPy implementation. It returns a new
vector with the same length, dtype and container type as y and never sorts
its inputs.

`check_increasing(x, y)` returns a Bool from Spearman's rank correlation.
The vectors must have equal lengths. Infinities participate in ranking. Empty
vectors, NaN values and constant inputs give undefined correlation and return
`False`. Direct calls write reference warnings to standard error as
`Category: message` lines, in occurrence order. A constant vector with two or
more entries triggers `ConstantInputWarning`. For more than three pairs, the
Fisher interval spanning zero or an undefined correlation triggers
`UserWarning`. Automatic fit records the same diagnostics in `fit_warnings`.

`score` returns NaN for a valid single-sample query and writes
`UndefinedMetricWarning` to standard error in the same format. It validates
score weights before issuing the warning. Moxie uses standard error for these
direct-call diagnostics in place of Python warning filters.

## Composition

- `clone(model)` and `model.copy()` follow the estimator rules above.
- `IsotonicCVAdapter[dtype]` and `IsotonicScorer[dtype]` work with
  `moxie.model_selection` `cross_val_score`, `cross_validate` and
  `GridSearchCV`. They use the estimator's NDArray input shapes. Set the
  second compile-time parameter `list_inputs=True` on both types for
  one-dimensional Lists. With Lists, predict or transform through
  `search.best_estimator_()`. The search object's forwarding methods use
  the estimator's NDArray trait types.
- `moxie.calibration` uses `IsotonicRegression` internally for
  `method="isotonic"`.
- `make_pipeline(StandardScaler(), IsotonicRegression())` accepts
  one-column X and a one-dimensional target array. Instantiate both steps
  with Float32 to use a Float32 pipeline.

Pass optional per-row weights explicitly as `fit_metadata` and `score_metadata`.
Each weight vector must align with the original samples and use the same dtype
and container as y. Moxie checks the full weight vector before selecting rows
for each fold. Scikit-learn can pass a mismatched-length vector through
unchanged to each fold.
The default split is an unshuffled five-fold `KFold`. To supply other splits, pass
a `List[IndexSplit]` through model selection's `cv` argument.

```mojo
from moxie.isotonic import IsotonicCVAdapter, IsotonicRegression, IsotonicSearchValue
from moxie.model_selection import GridSearchCV, cross_val_score

comptime Adapter = IsotonicCVAdapter[]
var model = IsotonicRegression(out_of_bounds="clip")
var adapter = Adapter()
var scores = cross_val_score[Adapter](
    model, X, y, adapter, adapter.default_scorer(), adapter.default_cv(X, y), None, None,
)
var grid = Dict[String, List[IsotonicSearchValue]]()
grid["y_max"] = [IsotonicSearchValue(8.0), IsotonicSearchValue(NoneType())]
var search = GridSearchCV[Adapter](model, grid, adapter)
_ = search.fit(X, y)
var predictions = search.predict(X)
```

Grid values use `IsotonicSearchValue`. Bounds take Float64 or `NoneType()`
to clear a bound. Direction takes Bool or `String("auto")`. Out-of-bounds
mode takes a String.

### Sample-weight requests

`set_fit_request(sample_weight=...)` and `set_score_request(sample_weight=...)`
mutate the model fluently. Pass `True` to request the standard name, an ASCII
identifier to request an alias, `False` to decline weights, or `None` to reject
weights supplied to a parent under the standard name. Both methods default
to rejecting incoming weights until configured. Omitting the argument in a
setter preserves that method's existing request.

```mojo
_ = model.set_fit_request(sample_weight="training_weights").set_score_request(sample_weight=True)
var requests = model.get_metadata_routing()
var weight_name = requests.fit.requested_name()
```

`get_metadata_routing()` returns an independent snapshot and satisfies
`FitScoreMetadataRequestProvider` from `moxie.metadata_routing`. Generic
parents can query both requests before or after fit. Fitted copies and
unfitted clones preserve both requests. Direct `fit` and `score` always use
their explicit `sample_weight` arguments. Pipeline and model selection still
require their existing explicit metadata arguments.

### Estimator tags

`get_tags(model)` from `moxie.base` returns a value snapshot with
`input_tags.one_d_array=True`, `input_tags.two_d_array=False`, and
`_skip_test=False` for both dtypes and in either fitted state. These are the
reference tag values. The estimator also accepts an `(n, 1)` array as a
single-feature input.

## Persistence

Use the experimental [native model format](../serialization.md) for fitted or
unfitted estimators. Select the schema with the same dtype as the model.

```mojo
from moxie.isotonic_serialization import IsotonicRegressionSchema
from moxie.serialization import load_model, save_model

comptime Schema = IsotonicRegressionSchema[]  # Float64
save_model[Schema](model, "isotonic.moxie")
var restored = load_model[Schema]("isotonic.moxie")
```

Use `IsotonicRegressionSchema[DType.float32]` for Float32. The schema stores
the current configuration separately from fitted state, including configuration
changed since the last fit. It restores independent threshold Lists without
refitting. Constructor values retain their normal fit-time validation.
Fit diagnostics describe the current instance's last successful fit.
They start empty after loading an archive.
Version 2 preserves the fit and score sample-weight requests for both fitted
and unfitted models. Version 1 archives remain readable and load with the
default requests.
Fitted thresholds must be finite, with strictly ascending X, monotone y in
the fitted direction, and a domain matching the two endpoints.

Loading binds the interpolator's raise behavior to the saved configuration,
as scikit-learn does when unpickling. For example, if you fit with `"raise"`
and then set `out_of_bounds="nan"`, the loaded model returns NaN outside the
training domain.

## Scope and validation

Python bindings, automatic heterogeneous dtype coercion, scalar sample
weights, pandas output and automatic metadata forwarding remain open.
Pass weights in the same dtype and container type as y.
`get_feature_names_out(input_features)` accepts and ignores the optional names,
as the reference does. Sparse inputs
are unavailable in this typed API; see [dense scope](scope.md). Errors use
Mojo `Error` with `ValueError`/`NotFittedError` prefixes. See the shared guide
for in-place fitting and typed configuration differences.

`pixi run python tools/isotonic_checks.py test` covers trait conformance,
validation, clone/copy ownership, repeat fits, 1-D and one-column equivalence,
strided arrays, direction, bounds, weighted R², and native persistence.
`pixi run python tools/isotonic_checks.py parity` compares 858 Float64 and
721 Float32 deterministic cases, including both public helpers with Lists and
NDArrays, default and labeled transform output, scaler pipeline predictions,
interpolator reconstruction, validation edges, warning
diagnostics, sample-weight requests and input tags, against the pinned Python
module and Cython tie engine. The scikit-learn
support wheel and SciPy come from the locked Pixi environment. Float64 uses
`atol=rtol=2e-12` and Float32 uses `atol=rtol=2e-6` to allow for differences
in rounding and reduction order.
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
