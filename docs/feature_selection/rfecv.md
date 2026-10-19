# Recursive feature elimination with cross-validation

`RFECV` selects a feature count by evaluating RFE elimination paths on held-out
folds, then repeats elimination on all training data and fits the final model.
It reuses the family RFE engine and SelectionState and the adopted shared CV
splitting, slicing and scoring contracts. Numerical fitting stays in the wrapped
estimator. Float32 and Float64 dense inputs are supported.

```mojo
from moxie.linear_model import Lasso
from moxie.feature_selection import RFECV, LassoImportance, DenseSupervisedCV

comptime CV = DenseSupervisedCV[LassoImportance[DType.float64]]
var selector = RFECV[CV](LassoImportance(Lasso(alpha=0.1)))
selector.fit(X, y)
var selected = selector.transform(X)
var predictions = selector.predict(X)
var results = selector.cv_results_()
var counts = results["n_features"][List[Int]].copy()
```

The [runnable example](../../examples/feature_selection/rfecv.mojo) also prints mean validation scores.
For Float32, use `DenseSupervisedCV[LassoImportance[DType.float32], DType.float32]`.
ElasticNetImportance works identically. Both adapters wrap real native models;
their solver restrictions remain applicable.

## Configuration and behavior

Constructor defaults match the reference: `step=1`, `min_features_to_select=1`,
`cv=None`, `scoring=None`, `verbose=0`, `n_jobs=None`, `importance_getter="auto"`.
The first argument is a typed SelectionEstimator; the compile-time CV adapter
must use the same estimator, input dtype and target type. RFE calls
`SelectionEstimator.fit_selection` for fitting; the CV adapter supplies row
slicing, labels and scorer configuration, and its `fit` method is not called.
An optional second
positional argument supplies a CV adapter instance.

`step` uses the same `FeatureCount` integer/fraction variant as RFE. Fractional
steps are computed once from the original width, rounded down, with a minimum
step of one. The last elimination clips the step to visit
`min_features_to_select` exactly. A minimum above the input width retains all
features and evaluates one point without emitting a Python warning.

`cv` uses the existing owned `SequentialCV` configuration: `SequentialCV(None)`
means five folds, an integer specifies the fold count, and `SequentialCV(pairs)`
snapshots shared `IndexSplit` pairs. Default folds are unshuffled KFold, or
StratifiedKFold for binary/multiclass classifier targets. For grouped, time-series
or custom splitting, materialize splits using the shared model-selection tools.
There is no new fold algorithm or stateful splitter-object interface here.

Each fold fits one descending RFE path with fresh estimator clones at every
step. Scores are summed across folds; the smallest feature count wins a tie.
NaN sums follow NumPy argmax: the first NaN in ascending feature-count order
wins. Exceptions from fitting or scoring propagate immediately. Full-data RFE
then selects the chosen count, and a further fresh estimator is fitted on those
columns, matching sklearn's final-refit lifecycle. No fold-fitted estimator is
published as `estimator_`.

`scoring` is an optional typed scorer instance. `None` asks the adapter for its
default scorer. The pinned sklearn implementation chooses accuracy for
classifiers and R2 otherwise. The supplied DenseSupervisedCV delegates to the
estimator's score, which is R2 for the supplied Lasso/ElasticNet adapters. Custom
adapters must supply the matching default metric; arbitrary estimator-specific
score overrides are not automatically translated into sklearn metric lookup.
`RFECV.score(X, y, sample_weight=None)` uses the current configured scorer on the
selected columns and final fitted estimator. Native scorers receive that adapter,
not the Python RFECV wrapper; custom wrapper-introspection semantics are not
supported. Prediction, probability, log-probability, decision and class-label
capabilities are forwarded only when the estimator adapter implements them.

`fit(X, y, sample_weight=weights)` routes training weights only.
`fit_named(X, y, names, sample_weight=weights, score_metadata=score_weights)`
accepts separate training/scoring weights. Both are optional rank-one arrays
with n_samples entries. Shared adapter methods slice metadata onto training and
test rows. The final estimator receives full training weights. Python routing
configuration and arbitrary keyword metadata are not exposed; custom adapter
metadata must also be optional input-dtype arrays.

Nonzero `n_jobs` values are accepted, but execution is serial, consistent with
the current native CV implementation. Sparse inputs/importances, Python string
scorer lookup, attribute-path reflection, implicit dtype coercion, container
output settings and warning transport remain outside this dense implementation.

## Learned state and result columns

Owned accessors expose `support_()`, `ranking_()`, `n_features_()`,
`n_features_in_()`, `feature_names_in_()`, `estimator_()` and `cv_results_()`.
Transforms, inverse transforms and feature names delegate to SelectorMixin and
SelectionState. Inputs may be strided; outputs and returned metadata are owned.

`cv_results_()` returns `RFECVResults`, a native dictionary whose values are
`RFECVResultColumn` variants. Keys retain sklearn names:

| Key | Native value and shape |
| --- | --- |
| `n_features` | `List[Int]`, ascending candidate counts |
| `mean_test_score`, `std_test_score` | Float64 NDArray, `(candidates,)`; population standard deviation |
| `split0_test_score`, etc. | Float64 NDArray, `(candidates,)` |
| `split0_support`, etc. | `List[List[Bool]]`, `(candidates, original_features)` |
| `split0_ranking`, etc. | `List[List[Int]]`, `(candidates, original_features)` |

For example, `results["mean_test_score"][nm.NDArray[DType.float64]]` accesses
scores and `results["split0_support"][List[List[Bool]]]` accesses fold masks.
This dictionary adds RFECV path metadata to the shared internal ScoreTable;
it is not a general CV result engine. Aggregation uses shared dense reductions,
including for a single candidate or a single materialized fold.

A successful refit replaces all learned state together. Failed fits preserve the
previous valid state. `get_params()` returns an independent typed RFECVParams
record; `set_params(record)` updates configuration without discarding fitted
state. `__sklearn_clone__()` recursively clones into an unfitted selector.
Ordinary copies preserve independent fitted state. Fitting mutates in place
without returning self. Empty feature-name lists mean unnamed input.

## Validation and limits

Run `pixi run build`, `pixi run test`, and `pixi run parity`. Both isolated
consumers fit public RFECV with a real model. Native contracts cover cloning,
repeat fits, failure atomicity through the final refit, metadata slicing,
stratified defaults, exact fit counts, custom scoring, ties, NaN/Inf scores,
owned nested results, feature names, optional capability forwarding and invalid
inputs. Classifier routing also has production DecisionTreeClassifier coverage with
strings and exact Int64 labels; see [classifier selection](classifier-selection.md).

`tests/feature_selection/parity_rfecv.py` loads the actual RFECV/RFE classes from local sklearn
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`, using installed sklearn
1.9.1 infrastructure and real Lasso/ElasticNet. Set `MOXIE_REFERENCE_PYTHON`
to that environment if the existing workspace oracle interpreter is unavailable.
The 68 real-model fixtures cover both dtypes, step/count grids, default and materialized CV,
negative strides, multi-output targets, distinct fit/scoring weights, custom
scoring and nonfinite scores. Masks, counts and all per-fold rankings compare
exactly. Default final scores compare with sklearn RFECV.score; custom final
scores compare with the scorer applied to sklearn's final estimator and selected
columns, matching the documented native adaptation. Raw wrapper scores are also
retained in the oracle output. Scores, coefficients and predictions use Float32 rtol=5e-5, atol=5e-6
and Float64 rtol=3e-8, atol=3e-9 to allow coordinate-descent and reduction
rounding on these bounded fixtures. These checks do not establish full sklearn
parity, arbitrary custom-estimator compatibility or performance claims.

Near-tied scores can select different feature counts across runtimes, especially
for Float32 models: native estimator scoring uses widened arithmetic, while
sklearn may round distinct path scores to the same Float32 value. RFECV compares
its actual score sums without an artificial tie tolerance. Exact-mask parity
fixtures use either controlled exact ties or separated score maxima; numerical
score tolerances alone cannot guarantee identical selections near a tie.
