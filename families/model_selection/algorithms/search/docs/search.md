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
comptime RegressionAdapter = DenseSupervisedAdapter[DummyRegressor[]]
var adapter = RegressionAdapter(configure)
var search = GridSearchCV[RegressionAdapter](model, grid, adapter)
_ = search.fit(X, y)
print(search.best_params_()["constant"])
var prediction = search.predict(X)

var random = RandomizedSearchCV[RegressionAdapter](
    model, ParameterGrid[Float64](grid), adapter,
    n_iter=3, random_state=42,
)
_ = random.fit(X, y)
```

The example's `configure` function applies each candidate dictionary to the
estimator's constructor parameters. It uses the same adapter for both searches
and for `cross_val_score[RegressionAdapter](model, X, y, adapter)`.

`DenseSupervisedAdapter[E, Y, dtype, V, weight_dtype, S]` works with ordinary
row-oriented NDArray inputs. It takes a concrete estimator type and the concrete
target type when it differs from Float64 NDArray. Specify the feature dtype
when it differs from Float64 and the weight dtype when it differs from the
feature dtype. Array targets support rank one or two with Float32, Float64
or Int64 elements. List targets support String, Int, Int64, Float32 or Float64.
Each fold preserves target layout and dtype.

Use the same adapter for out-of-fold predictions when the estimator returns
one of those array or list types. The result keeps the estimator's prediction
type, dtype and output shape. The optional `fit_metadata=` passes a weight
array to fit, and `cv=` overrides the adapter's default folds. Use the
[class prediction adapter](../../validation/docs/scope.md#out-of-fold-predictions)
for probabilities or decision values.

```mojo
var predictions = cross_val_predict[RegressionAdapter](model, X, y, adapter)
```

The adapter calls the estimator's default score unless you supply a scorer type
as `S`. A custom scorer must match the estimator, input, target, and optional
weight type. The adapter constructs a defaultable scorer or accepts an instance
with `scoring=`. The selected scorer works for validation and search.

```mojo
comptime Model = DummyRegressor[]
comptime Dense = NDArray[DType.float64]
comptime MAE = NegMeanAbsoluteError[Model, Dense]
comptime MAEAdapter = DenseSupervisedAdapter[Model, S=MAE]
var scores = cross_val_score[MAEAdapter](model, X, y, MAEAdapter())
```

Classification scorers `Accuracy`, `PrecisionMacro`, `RecallMacro`, and
`F1Macro` select the corresponding scikit-learn scorer by Mojo type. For SVC
with its default Float64 labels:

```mojo
comptime Model = SVC[]
comptime Labels = List[Float64]
comptime Scorer = PrecisionMacro[Model, NDArray[DType.float64], Labels]
comptime Adapter = DenseSupervisedAdapter[Model, Labels, S=Scorer]
var scores = cross_val_score[Adapter](SVC(), X, y, Adapter())
```

The scorer's prediction and target types must agree. Its optional score-weight
dtype defaults to Float64. Use `weight_dtype=DType.float32` for Float32 weights
and match that dtype in the adapter. The underlying metrics handle class
selection and undefined scores.

The optional
`apply_params` argument accepts a noncapturing Mojo function with signature
`def(mut E, Dict[String, V]) thin raises`. `V` defaults to Float64. An empty
parameter dictionary evaluates the original configuration. Nonempty candidate
dictionaries require an applier that rejects unsupported names.
Fit and score metadata are separate optional rank-one weight arrays. Weighted
fit requires `WeightedSupervisedEstimator`. The configured scorer receives
score weights.

Omitting `cv` generates unshuffled folds from the current fit data.
The dense adapter defaults to five folds. Set `n_splits=` on the adapter to
choose another count while generating fresh folds on each fit.
Classifiers use stratification for binary and multiclass targets. An estimator
wrapper that exposes a classifier's score without implementing `ClassifierMixin`
can choose the classifier policy with `classifier=True`, as in the
[StandardScaler/PCA/SVC fixture](../../../../../tests/pipeline/test_nested_pca.mojo).
Use a custom adapter for pairwise kernel slicing or other metadata layouts.

For a Float32 or Float64 estimator implementing `UnsupervisedScorer`, use
`DenseUnsupervisedSearchAdapter[Estimator, dtype]`. Pass a typed function
that applies a candidate's parameter dictionary and rejects unknown keys.
For example, with a Float64 grid of `ShrunkCovariance` shrinkage values:

```mojo
comptime Adapter = DenseUnsupervisedSearchAdapter[ShrunkCovariance[]]
var search = GridSearchCV[Adapter](ShrunkCovariance(), grid, Adapter(apply_shrinkage))
_ = search.fit(X, None)
```

The adapter uses ordinary sample rows, omits the estimator's target and calls
its `score(X)`. Like `cross_val_score(estimator, X)`, it defaults to five
unshuffled folds. Set `n_splits=` on the adapter or pass explicit `cv=` splits
to the search constructor. The same adapter works with `RandomizedSearchCV`.
See the [covariance notebook](../../../../../notebooks/covariance/plot_covariance_estimation.ipynb)
for the complete grid and parameter applier.

## Adapter and configuration contracts

`SearchAdapter` composes `DefaultCVAdapter` and the family `ParameterAdapter`.
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
- `default_cv(X, y)`, supplying unshuffled folds for `cv=None`: stratified
  for binary/multiclass classification, ordinary KFold otherwise. The dense
  adapter defaults to five.

Search adapters and their scorers are copyable. Inputs, targets, metadata and
estimator parameter records must be deinitializable. The wrapped estimator needs
`BaseEstimator` and cloning. Prediction and fitted-state copying depend on the
wrapped estimator’s capabilities. The explicit `fit` overload takes separate fit/score metadata;
the two-argument overload constructs them when their type is `Defaultable`.
Use `None` or an explicit empty record when unused. The inherited adapter slices
sample-aligned weights and preserves unsliced options. Pairwise inputs require
a pairwise-aware adapter. These responsibilities are not inferred dynamically.

`cv` accepts an optional materialized `List[IndexSplit]`. Supply explicit folds
for grouped, temporal or custom CV. Constructors and
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

The dense adapter's focused test compares default KFold indices, weighted
Float32 regression scores and the full-data refit with the pinned reference.
Run it from the repository root with the locked environment and assembled
sources.

```sh
mojo build -O3 --fp-mode=contract=off -I builds/deps \
  families/model_selection/algorithms/search/tests/dense_adapter.mojo \
  -o builds/dense_adapter
python families/model_selection/algorithms/search/tests/dense_adapter_parity.py \
  builds/dense_adapter
```

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
Both searches expose `classes_()` for estimators implementing
`NumericClassifierClasses` and `n_features_in_()` for estimators implementing
`InputFeatureCount`. These accessors read the refitted winner and raise
`AttributeError` when there is no refitted winner. `classes_()` returns an
owned label array or list in fitted class order. Searches also forward the
metadata traits to generic callers. Numeric class trait conformance requires
defaultable scorer metadata and a scorer target type matching the estimator's
target. Read other learned attributes through `best_estimator_()`.
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

## Composition

Both searches can be the estimator in a concrete CV scorer.
`TypedScorer` and its input, target and metadata are deinitializable. Search
also checks at compile time that its scorer is copyable. The shared
`BaseEstimator.Params` contract supplies parameter ownership.

Every search `fit` overload returns a mutable borrow of the search, including
random search's application-owned RNG overload. `fit(X, y)` constructs separate
default fit and score metadata values when `Metadata` is `Defaultable`. Only
those searches conform to `SupervisedEstimator`; metadata with a required
constructor remains explicit. A defaultable metadata type must represent a
valid empty request if it is intended for this convenience overload.

With defaultable metadata, searches also forward the underlying estimator's
classifier/regressor kind and supervised scoring capability. `score(X, y)` uses
the selected refit scorer, which may differ from accuracy or R2 when configured,
as in sklearn. `score(X, y, metadata=request)` supplies typed score metadata.
The metadata keyword is required to distinguish it from the shared
`sample_weight` argument, whose type may be identical. A nonempty
`sample_weight=` raises with guidance to use metadata; no routing is inferred.
This keyword requirement replaces the old positional metadata call.

A search is copyable when its estimator is copyable. A copy preserves fitted
state with independent buffers; `clone(search)` retains only constructor
configuration. A supervised search with prediction support can be a
Pipeline’s final step.

Inner folds that must follow each training subset belong in the adapter's
`default_cv`, because `cv` holds materialized indices for one sample count.
From the repository root, use the locked environment to assemble live
sources and run the Pipeline and nested SVC comparisons.

```sh
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mkdir -p builds/search-composition
for name in pipeline_model_selection nested_search nested_pca; do
    mojo build -O3 --fp-mode=contract=off -I builds/deps \
        "tests/pipeline/test_$name.mojo" -o "builds/search-composition/$name"
done
python tests/pipeline/composition_parity.py builds/search-composition
```

The comparison loads the pinned sklearn search, validation and Pipeline sources.
It covers StandardScaler/SVC, PCA/SVC, StandardScaler/PCA/SVC, nested SVC search
and search as a Pipeline final step. Candidate order, folds, ranks, best
selections and refit predictions match exactly. Float64 candidate means and
the weighted `EstimatorScore` allow an absolute tolerance of 1e-12 for
reduction rounding.

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
