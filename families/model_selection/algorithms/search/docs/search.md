# Grid and randomized search

`GridSearchCV` and `RandomizedSearchCV` evaluate native estimators through the
family's CV adapters. Execution is serial. Every candidate uses the same owned
folds, every fold fits an independent clone, and successful refitting trains a
fresh winning configuration on the full input. Python is used only by tests.

The runnable [example](../examples/search.mojo) tunes a real DummyRegressor:

```mojo
var grid = Dict[String, List[Float64]]()
grid["constant"] = [0.0, 2.0, 4.0, 5.0, 9.0]
var model = DummyRegressor(strategy="constant")
var search = GridSearchCV[RegressionAdapter](model, grid, RegressionAdapter())
search.fit(X, y, None, None)
print(search.best_params_()["constant"])
var prediction = search.predict(X)

var random = RandomizedSearchCV[RegressionAdapter](
    model, ParameterGrid[Float64](grid), RegressionAdapter(),
    n_iter=3, random_state=42,
)
random.fit(X, y, None, None)
```

Run `pixi run test` in `algorithms/search` to compile and run the example.
The example adapter is explicit so users can adapt it to their estimator's
parameter, target and metadata types. Dummy estimators are live test/example
dependencies from Moxie core (`moxie.dummy`); the search implementation imports no concrete estimator.

## Adapter and configuration contracts

`SearchAdapter` composes `CVAdapter` and the family `ParameterAdapter`.
Its existing adapter implementations also work with `validation_curve`; no
search import is required by the validation engine. Its contracts are:

- `Value`, the copyable, deinitializable value type for candidate dictionaries.
  Use a tagged record for heterogeneous values. Distributions and the adapter
  must agree on this type; the compiler rejects mismatches.
- `configure(mut estimator, params)`, applying a partial constructor-parameter
  dictionary. Reject unknown names. Clone nested estimator configurations rather
  than retaining fitted state or caller aliases.
- `default_scorer()`, supplying the wrapped estimator's default score when
  `scoring=None`. A provided scorer can use any supported estimator capability.
- `default_cv(X, y)`, supplying five unshuffled folds for `cv=None`: stratified
  for binary/multiclass classification, ordinary KFold otherwise.

Adapters and scorers are copyable. Inputs must be copyable and deinitializable;
targets, metadata and estimator parameter records must be deinitializable. The
wrapped estimator needs `BaseEstimator` and cloning, not fitted-state copying
or prediction. Fit/score metadata are separate required arguments to `fit`;
use `None` or an explicit empty record when unused. The inherited adapter slices
sample-aligned weights and preserves unsliced options. Pairwise inputs require
a pairwise-aware adapter. These responsibilities are not inferred dynamically.

`cv` accepts an optional materialized `List[IndexSplit]`. Supply explicit folds
for a different fold count, grouped, temporal or custom CV. Constructors and
parameter records own their indices; mutating the original arrays cannot change
search folds. Fold order and duplicate/overlapping custom test indices are
preserved. Fold means are unweighted even when fold sizes differ.

`scoring` accepts an optional list of one scorer type. A tagged scorer can
represent different metric implementations. Omit `metric_names` for one scalar
score. Supplying names explicitly selects multi-metric behavior, even for one
name; names must be unique, nonempty and aligned with the scorers. One estimator
fit serves all metrics. Scores must already use the higher-is-better convention.

Grid search accepts a dictionary, a list of dictionaries, or `ParameterGrid`.
Randomized search accepts `ParameterGrid` or lists of typed `ParameterDomain`
spaces, with `n_iter=10` by default. List-only spaces sample without replacement,
capped at grid size; any distribution switches to sampling with replacement.
Both use the existing candidate/RNG implementation.

An integer `random_state` reconstructs the sampling stream for every fit and
clone. The default `None` requires the overload
`fit(X, y, fit_metadata, score_metadata, rng)` with an application-owned mutable
`RandomState`. That stream advances. Providing both a configured integer seed
and a stream raises. There is no hidden global RNG.

## Results and refitting

Learned values use checked accessors ending in `_()`:
`cv_results_`, `best_index_`, `best_params_`, `best_score_`, `best_estimator_`,
`n_splits_`, `multimetric_`, `scorer_`, and `refit_time_`.

`cv_results_()` returns an independent `SearchResults[V]` snapshot:

| Field | Layout and meaning |
| --- | --- |
| `params` | Candidate dictionaries in evaluation order |
| `scores` | `ScoreTable`: Float64 `(candidate, split, metric)` test scores and optional train scores; ordered metric names |
| `status` | Candidate-major `FitStatus` records with fit failures, diagnostics and times |
| `mean_test_score`, `std_test_score`, `rank_test_score` | `(candidate, metric)`, population standard deviation and minimum ranks for ties |
| `mean_train_score`, `std_train_score` | Optional `(candidate, metric)` arrays; default absent |
| `mean_fit_time`, `std_fit_time`, `mean_score_time`, `std_score_time` | Float64 `(candidate,)`, seconds |
| `parameter(name)` | One optional value per candidate; missing conditional-grid parameters are `None` |

Means and deviations are Float64. Ranks use the family's Int64 index convention
(the reference rank columns are Int32).

NaNs propagate through moments; ranking follows the reference's actual
`nanmin - 1` replacement and clipping of original infinities to finite extrema,
including infinity and large-number rounding edges.
All-NaN means tie at rank one. The first candidate with the best rank wins.
`return_train_score=False` is the default.

`refit=True` is the default. Pass `False` to skip refitting or a metric name for
multi-metric selection (`SearchRefit[V](...)` is the explicit typed spelling).
Single-metric `refit=False` still exposes the best index, parameters and score;
multi-metric `refit=False` exposes none of those. Inference requires refitting.

For custom selection, implement `SearchRefitPolicy`, set `Value` and
`Results = SearchResults[Value]`, supply a default constructor and
`select(results) -> Int`, and pass the policy as a compile-time type:
`GridSearchCV[Adapter, Policy](..., refit=Policy(...))` or
`RandomizedSearchCV[Adapter, Distribution, Policy](...)`. The policy receives all
result fields. An out-of-range index raises; custom refitting omits `best_score_`.
Use `score_metric(name, X, y, metadata)` for custom multi-metric refitting, which
has no implicit score metric. Ordinary `score` uses the configured scalar or
selected named scorer, not an unrelated default estimator score.

The winner is borrowed through `best_estimator_()`; no fitted copy is fabricated.
Read wrapped learned metadata such as `classes_` and `n_features_in_` there.
Prediction, probability, log-probability, decision, transform, inverse-transform
and sample-score methods delegate only when the wrapped estimator implements
the corresponding capability. Delegated inputs must match the estimator's input
type. The inverse-transform method is available conditionally, but these search
structs do not advertise the `InverseTransformer` trait on Mojo 1.0 because that
trait couples the inverse output to the independent scorer input type.

`get_params(deep=True)` returns a typed `SearchParams` record with constructor
`estimator` parameters, `space`, and `options`. `set_params(params)` replaces
configuration and clears learned state. Named overloads update `param_grid` or
random search's `param_distributions`, `n_iter`, and `random_state`. Clones own
unfitted constructor configuration. A new fit first clears learned state;
failed repeated fits and failed final refits cannot expose an old winner.

## Composition limits

`fit(X, y, fit_metadata, score_metadata)` mutates in place and returns None,
unlike the shared fluent `fit` contract, and the explicit metadata arguments
mean the search structs do not conform to `SupervisedEstimator`. A search
therefore cannot be the final step of `moxie.pipeline.Pipeline`.

On Mojo 1.0 a struct with a struct-level `where` clause cannot be named as a
parameter, field or associated type outside its own generic context: the
compiler reports "cannot implicitly convert 'GridSearchCV[...]' value to
'GridSearchCV[...]'" or crashes. `GridSearchCV`, `RandomizedSearchCV` and
`moxie.pipeline.Pipeline` all carry such clauses, so none of them can be a
scorer's `Estimator`: `cross_validate`, `cross_val_score`, the curves, the
permutation test and the searches do not accept a search or a pipeline as the
evaluated estimator. Write nested cross-validation as an explicit loop: `clone`
the search per outer fold, `fit` on the outer training rows and `score` the
outer test rows (see `notebooks/model_selection/plot_nested_cross_validation_iris.ipynb`).
Inner folds that must follow each training subset belong in the adapter's
`default_cv`, because `cv` holds materialized indices for one sample count.

## Failures and supported scope

Default `error_score` is `ErrorPolicy(False, NaN)`; pass
`ErrorPolicy(True, 0)` to propagate errors or choose a numeric replacement.
Fit failures fill every metric for that candidate/fold; scorer failures fill
only that metric. A candidate failing every fold does not abort successful
candidates. All search fits failing raises. Final refit always propagates errors,
even with a numeric replacement. Configuration and slicing errors also propagate.
Fit/scoring exceptions retain owned diagnostics. Python warning emission is
unavailable.

Timing retains the CV engine's native convention: fit time measures adapter fit,
excluding slicing and cloning; score time measures test scoring. Failed fits have
zero score time. Timing values are checked for validity, not equality with Python.

Sparse inputs/outputs remain deferred without densification. Python reflection,
scorer-name lookup, `n_jobs`/`pre_dispatch`, verbosity, callbacks and metadata
routing are not exposed. Typed configuration, accessors, explicit adapters and
stream ownership are documented Mojo interface differences. This is dense serial
search support, not complete Python/scikit-learn interoperability.

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity` in this child, plus
root `pixi run build` and `pixi run test`. The root compiles and runs a public
search consumer; both isolated namespaces assemble the new exports.

Tests use the real live DummyRegressor/DummyClassifier for regression,
classification, weights, default stratification, probability delegation and
refitting. Capability-only probes check one fit per metric set and nonpredictive
estimators; they are explicitly test doubles. Negative compilation rejects a
probability method on a regressor without that capability. Lifecycle checks
cover cloning, parameter changes, failed/repeated fits, custom refit, missing
parameter fields, empty candidates/folds, scorer failures and RNG replay.

The oracle loads original search classes and CV function bodies at sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with sklearn 1.9.1 supplying supporting
validation, callbacks and tags. Real pinned Dummy classes supply the estimators.
The default reference interpreter is Moxie's Pixi Python under the Moxie root `.pixi` directory;
`MOXIE_REFERENCE_PYTHON` can select another with sklearn 1.9.1. Candidate/RNG parity
is exact. Search scores/moments/predictions use `rtol=atol=1e-12` for bounded
Float64 fixtures to allow reduction rounding; ranks and winning indices are
exact. Nonfinite rank fixtures, weighted default scores and classification folds
are also compared. These tests do not certify every estimator, Float32 numerical
kernel, metadata combination or large/ill-conditioned dataset. No performance
claim is made.
