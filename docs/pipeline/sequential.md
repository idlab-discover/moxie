# Typed sequential pipelines

`moxie.pipeline.Pipeline` now executes a transformer followed by a supervised
predictor in pure Mojo. It accepts existing foundation capability traits without
an estimator-specific adapter or `CompositionStep` conformance. Components move
into the pipeline; their parameters and learned state remain owned by them.

```mojo
from moxie.pipeline import Pipeline
from moxie.base import clone

# transformer and model are concrete estimators supplied by their families.
var pipe = Pipeline(transformer^, model^,
                    transformer_name="prepare", estimator_name="model")
pipe.fit(X, y)
var prediction = pipe.predict(X_test)
var fresh = clone(pipe)  # child clone hooks, no copied learned state
var params = fresh.get_params()
params.estimator = model_params
fresh.set_params(params)
```

The prefix must implement `TransformerMixin` and either `FitTransformer`,
`SupervisedEstimator`, or `UnsupervisedEstimator`. Its `fit_transform` is preferred;
otherwise execution calls `fit` and then `transform`. The target is forwarded to
every fit. The final estimator implements `SupervisedEstimator` and `Predictor`.
Intermediate output and successor input types must match exactly. When an
unsupervised prefix (`UnsupervisedEstimator`, optional target) declares a
different target type than the final estimator, as `StandardScaler`
(`NDArray`) before `SVC` (`List[Label]`) does, the prefix is fitted with
`y=None`. sklearn passes `y` to such steps and they ignore it, so results are
identical; a prefix that actually needs `y` raises its own missing-target error.
Otherwise target types must match at compile time. Targets must be copyable and destructible, and intermediate results and
constructor parameter records must be destructible. Optional-target foundation
methods currently require an owned target copy; no implicit dtype conversion
occurs. Inputs are borrowed and intermediate results live until the next method
returns. The engine introduces no numerical operations or array implementation.

Longer sequences use nested pipelines:

```mojo
var tail = Pipeline(second_transformer^, model^)
var pipe = Pipeline(first_transformer^, tail^)
```

This is nested sklearn Pipeline composition, not a flattened heterogeneous list.
Names are unique within each container. Names, child instances, and typed child
parameters are available as `transformer_name`, `estimator_name`, `transformer`,
and `estimator`. Assigning a same-type child replaces that instance, preserving
its state; changing a static type requires constructing a new pipeline.

`PipelineParams` holds both names and both child parameter records. `deep` is
forwarded to each child; the native record shape is unchanged for `deep=False`.
Name validation precedes all parameter updates. Child updates run in order and
are not transactional if a child raises. Constructor name validation is eager
(unlike sklearn's fit-time validation), and fit validates again after public
field edits. Reserved names match sklearn's constructor parameters. This API
uses two typed arguments instead of Python's list of `(name, estimator)` pairs.

Normal fit uses the stored instances directly, corresponding to `memory=None`.
Repeated and failed fits preserve each child's own reset rules. The pipeline's
fitted query delegates to its final estimator. A failed prefix refit can therefore
leave that query true while prediction fails in the prefix, as in the reference.
Clone delegates to each child hook, including any documented frozen-estimator
exception. The pipeline does not cache a separate fitted flag.

## Feature names

`get_feature_names_out(input_features=None)` forwards names through the prefix
and then the final estimator, including nested pipelines. Each step must opt
into the shared `UnionFeatureNames` trait (also used by
FeatureUnion). Its signature is
`get_feature_names_out(self, input_features: Optional[List[String]] = None)
raises -> List[String]`. The historical name remains an alias of `moxie._feature_metadata.FeatureNamesOut`.
`ColumnTransformer` implements this optional-name contract directly. `None` uses
learned/default names; an explicit empty list is validated as supplied.

```mojo
# Both children must provide UnionFeatureNames.
var input_names: List[String] = ["height", "width"]
var output_names = pipe.get_feature_names_out(input_names.copy())
var default_names = pipe.get_feature_names_out()
```

Children determine default names, output ordering, and fitted/input validation.
Pipeline does not add step prefixes, remove duplicate names, or impose a fitted
check of its own. Empty lists are forwarded as supplied. Missing capabilities
raise an AttributeError-style error identifying the step; earlier child errors
propagate first. A final predictor without feature names therefore raises.
The current API has no slicing: request prefix names directly from
`pipe.transformer.get_feature_names_out(...)` when only those are needed.

The native result is an owned `List[String]`, rather than NumPy's string array;
input names are borrowed and are not mutated by the container. Native contracts
and seven exact-string comparisons against local scikit-learn cover explicit,
default, empty and duplicate names, nesting, cloning and refitting, with separate
checks for unsupported children and child validation failures.

## Input feature count

`pipe.n_features_in_()` delegates to the first step through the existing
`UnionInputFeatures` trait. Like FeatureUnion, this is a method in Mojo rather
than a Python property. Children expose `n_features_in_(self) raises -> Int`.
The value is the first step's input width, even when that step changes the
number of columns. A FeatureUnion prefix uses its own first-branch rule.

For constituents with a stored attribute, `InputFeatureCount` from
`moxie._feature_metadata` instead exposes `get_n_features_in(self) raises -> Int`.
`ColumnTransformer` implements it while retaining its public `n_features_in_`
attribute; Pipeline and FeatureUnion recognize both count contracts.

No count or fitted flag is cached by Pipeline. The first step decides whether
the count is available; it can remain available while the final estimator is
unfitted. Refitting, replacing the prefix, and child failures therefore take
effect immediately. A prefix without either feature-count capability raises an explicit
AttributeError-style error naming that step. Clone follows the child's clone
semantics, normally leaving the count unavailable until fit.

Float32/Float64 tests cover a dimensionality-changing fixture, cloning, refits,
child replacement and failure, unsupported prefixes, and a FeatureUnion prefix
followed by a nested Pipeline with the real DummyRegressor. Six exact count
comparisons run against the pinned scikit-learn implementation.

## Scoring

`pipe.score(X, y, sample_weight=None)` is available when the final estimator
implements the foundation `SupervisedScorer` trait. Such a Pipeline also conforms
to `SupervisedScorer`, including when the final estimator is another Pipeline.
The target and weight types come from the final estimator. Unlike Python's
optional `y`, this supervised API requires `y`, following the foundation trait.

```mojo
var score = pipe.score(X_test, y_test)
var weighted_score = pipe.score(X_test, y_test, sample_weight=weights.copy())
```

Scoring checks pipeline fitted state, transforms `X` through the prefix, and
delegates directly to the final estimator. It does not fit any step or call
`predict`. Targets and weights are forwarded unchanged and only the final
estimator receives weights. The child owns score semantics, including accuracy
or R², validation, and configuration-dependent errors. Pipeline does not claim
`ClassifierMixin` or `RegressorMixin` conformance merely because it can score.

Unsupported final estimators fail at compile time; ordinary fit/predict remains
available. The foundation scoring trait includes optional weights, so a native
scorer that cannot accept them must reject nonempty weights explicitly. Python
reflection, arbitrary score metadata, and optional-target/unsupervised scoring
remain unsupported. Weighted scoring does not enable weighted fitting.

Tests compare four exact scalar delegation cases and real DummyRegressor R²
with and without weights in Float32/Float64 against local scikit-learn. The
dense fixtures yield exactly representable scores. Native contracts also cover
unfitted/failed-fit behavior, clone and parameter updates, invalid targets and
weights, conditional trait dispatch, and missing-score compilation failure.

## Probability prediction

`predict_proba(X)` is available when the final estimator implements
`ProbabilityPredictor`. `predict_log_proba(X)` separately requires
`LogProbabilityPredictor`; probability support alone does not enable it.
Pipeline conditionally conforms to those same foundation traits, so both
methods work through nested pipelines and generic capability dispatch.

```mojo
var probabilities = pipe.predict_proba(X_test)
var log_probabilities = pipe.predict_log_proba(X_test)
```

Each call checks fitted state, transforms the prefix, and directly invokes the
corresponding final method. Pipeline never computes logarithms, calls `predict`,
normalizes probabilities, sorts classes, or fits steps during inference. The
final estimator determines output type, shape, dtype, ordering, and validation.
Class labels remain available on the final estimator; forwarding `classes_`
is not yet implemented. Unsupported methods fail at compile time, and child
errors propagate. Extra prediction arguments and metadata routing remain absent.

Native tests and eight reference cases cover independent capabilities, nested
and fused-prefix dispatch, cloning, refits, failed fits, and real DummyClassifier
integration for Float32/Float64 inputs. Real-estimator checks include sorted
class columns, output independence, zero probabilities with negative-infinite
logs, and single-class refits. This adopted DummyClassifier returns Float64
probabilities for both input dtypes; Pipeline preserves that type. Probabilities
compare exactly on these fixtures; logarithms use rtol/atol `1e-12` to allow
native math rounding. Negative infinity must match exactly.

## Decision values

`pipe.decision_function(X)` is available when the final estimator implements
the foundation `DecisionFunction` trait. Pipeline then conforms to that trait
as well, including in nested pipelines. It checks fitted state, transforms the
prefix, and returns the final estimator's decision output directly.

The child determines the result type, rank, dtype, sign convention, and column
order. Pipeline does not call `predict`, convert decisions into probabilities,
or assign classifier semantics. Unsupported final estimators fail at compile
time; child errors propagate. Extra method arguments and metadata routing are
not supported.

Eight exact reference comparisons use explicit dispatch/shape fixtures: four
scalar cases and four dense cases spanning Float32/Float64 and rank-1/rank-2
outputs. Native contracts additionally cover cloning, parameter changes,
unfitted and failed-fit behavior, invalid inputs, empty output rows, independent
result storage, and conditional capability dispatch. These fixtures validate
composition; no new decision-producing statistical estimator is implemented.

## Per-sample scores

`pipe.score_samples(X)` is available when the final estimator implements
`SampleScorer`. Pipeline conditionally conforms to that foundation trait, so
the method also works through nested pipelines and generic trait dispatch.
After checking fitted state, it transforms the prefix and directly returns the
final estimator's scores, preserving their type, dtype, shape, order, and signs.
It does not aggregate scores, call `predict`, or infer density/outlier semantics.

Unsupported final estimators fail at compile time; child errors propagate.
This addition retains the existing supervised-predictor final-step requirement.
It does not yet enable unsupervised-only estimators such as LOF or KernelDensity
as final steps. `score_samples` support also does not imply `score` support.

Six exact comparisons with local scikit-learn cover scalar dispatch, cloning
and refits, fused prefixes, nesting, and dense Float32/Float64 sample order.
Native contracts additionally cover failed fits, invalid shapes, empty rows,
output ownership, and capability rejection. These are explicit composition
fixtures, not implemented density or outlier algorithms.

## Current boundary

This implementation exposes fit, predict, conditional score, per-sample scores, decision and probability
methods, feature names, input feature counts, configuration, and cloning.
It does not yet provide transformer-final/unsupervised-final pipelines,
inverse transforms, fit sample-weight routing, passthrough
sentinels, indexing, other learned attribute forwarding, automatic step naming,
metadata routing, caching, callbacks, output-container selection, or verbose
execution. These methods/settings are absent from the typed API.
`make_pipeline` and `make_union` remain explicit error-raising scaffolds.
[Dense FeatureUnion](feature-union.md) is implemented and can be a Pipeline prefix. Sparse execution remains deferred. This is not full sklearn parity
and now owns Moxie's pipeline module.

## Validation and reuse

Run `pixi run test` and `pixi run parity`. The test task builds all modules and
runs the inherited contracts, two isolated consumers, sequential execution and
real-estimator integration, plus negative capability/type and ownership checks.
The integration consumes the implemented `moxie.dummy.DummyRegressor` from Moxie's own sources, in Float32 and Float64. Its copy transformer is a
named test fixture, not an implemented preprocessing estimator. Scalar probes
separately establish order-sensitive behavior that DummyRegressor cannot test.

Parity executes Pipeline and DummyRegressor from the local scikit-learn revision
recorded in [interfaces](../interfaces.md); the locked sklearn wheel supplies supporting utilities
and base classes. Seven sequential cases cover dispatch, nesting, clone/refit and
failure behavior. Dense tests cover mean, median, clone independence and refit in
both dtypes. Small integer fixtures and their means/medians are exactly
representable, so equality is exact. This evidence does not certify arbitrary
estimators, numerical edge cases, or all sklearn pipeline behavior.
`builds/pipeline-parity.json` records source hashes and comparison results.

Moxie owns the pipeline facade, execution engines, and plumbing. Its
`DEPENDENCIES.json` assembles one namespace from current local sources; building
Moxie no longer requires the `moxie-pipeline` checkout. See
[composition contracts](plumbing.md) for extension and validation details.
