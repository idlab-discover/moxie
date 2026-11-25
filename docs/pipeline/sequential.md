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

`len(pipe)` counts the two immediate steps, including in a nested pipeline.
Use `pipe.step[0]()` or `pipe.step[-2]()` for the prefix and
`pipe.step[1]()` or `pipe.step[-1]()` for the final estimator. The index
is a compile-time parameter because it determines the return type.
An out-of-range index fails compilation.

For a name known at run time, use
`pipe.named_step[StandardScaler[]]("standardscaler")` with the expected
concrete child type. An unknown name raises a KeyError-style error and a
known name with the wrong type raises a TypeError-style error. Both accessors
borrow the stored child immutably, before or after fitting. The public child
fields support mutation. `FrozenEstimator` follows the same
[access contract](../frozen/api.md#access-to-pipeline-steps).
The [step parity check](../../tests/pipeline/steps_parity.py) compares
lengths, step selection, slices and fitted outputs with pinned scikit-learn.

Use `pipe.slice[stop=-1]()` for the prefix, `pipe.slice[-1]()` for the final
step, and `pipe.slice[]()` for both. Bounds are compile-time, half-open and
clipped like Python slice bounds. Only stride 1 is accepted. Slices count
immediate steps, so a nested pipeline remains one step at its parent level.

A slice borrows its source and provides read-only access to the selected fitted
steps. Its `step` and `named_step` accessors work within the selected range.
Nonempty slices forward `transform`, `predict`, `predict_proba` and
`get_feature_names_out` when the selected estimator supports them.
`view.estimator()` borrows that estimator for other read-only capabilities.
An empty slice has length zero and reports fitted, with no inference methods.
The source must outlive use of its slice. The view observes a later fit on the
source, and its step accessors borrow children immutably. In sklearn, a slice
can also mutate the shared children directly.

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
When both steps implement `Copyable`, `pipe.copy()` copies both child
instances with their fitted state. Pipelines with a move-only step still support
configuration cloning through the child's clone hook.

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
Request names for just the prefix with
`pipe.slice[stop=-1]().get_feature_names_out(...)` when the prefix supports
feature names.

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
attribute. Pipeline and FeatureUnion recognize both count contracts and also
implement `InputFeatureCount` themselves, so `FrozenEstimator` can read their
input counts through `get_n_features_in()`.

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
or R², validation, and configuration-dependent errors. Pipeline conforms to
`ClassifierMixin` when its final estimator does, including through nesting.
This lets classifier adapters consume the entire pipeline.

For a final scorer that implements `FitScoreMetadataRequestProvider` and has
a copyable weight type, use the explicit routing overload:

```mojo
_ = pipe.estimator.set_score_request(sample_weight=String("validation_weight"))
var score = pipe.score(X_test, y_test, metadata={"validation_weight": weights.copy()})
```

The dictionary uses the final scorer's weight type. An empty dictionary uses
ordinary unweighted scoring. The pipeline validates supplied keys before
transforming `X`. The standard key raises `UnsetMetadataPassedError` for an
unset request. Declined, unknown, or extra keys raise `TypeError`. The explicit
dictionary carries only supplied keys, so an alias works without an implicit
`sample_weight=None` argument. Routing covers the final supervised scorer.

Unsupported final estimators fail at compile time; ordinary fit/predict remains
available. The foundation scoring trait includes optional weights, so a native
scorer that cannot accept them must reject nonempty weights explicitly. Python
reflection, arbitrary score metadata, and optional-target/unsupervised scoring
remain unsupported.

Tests compare four exact scalar delegation cases and real DummyRegressor R²
with and without weights in Float32/Float64 against local scikit-learn. The
dense fixtures yield exactly representable scores. Native contracts also cover
unfitted/failed-fit behavior, clone and parameter updates, invalid targets and
weights, conditional trait dispatch, and missing-score compilation failure.

## Final-step sample weights

When the final estimator conforms to `WeightedSupervisedEstimator`, the
pipeline also conforms to that trait and accepts
`pipe.fit(X, y, sample_weight=weights)`. This keyword takes the final
estimator's weight type and replaces sklearn's `stepname__sample_weight`
spelling for the terminal step. Nested pipelines forward the same argument
through their final estimators. The prefix fits with its ordinary inputs and
target, and the final estimator validates the borrowed weights. Omit the
keyword for unweighted fitting. The weighted overload is unavailable at
compile time for other final estimators.

Float32 and Float64 tests use StandardScaler and DummyRegressor to compare
direct and nested weighted fits and an unweighted refit against pinned
scikit-learn. Predictions agree exactly on these fixtures. The unweighted
scaler output is compared within `2e-6` for Float32 and `1e-12` for Float64 to
allow rounding differences in standardization. Native tests also check
generic trait dispatch, cloning, borrowed inputs, and invalid weights.

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
Unsupported methods fail at compile time, and child errors propagate. Extra
prediction arguments and metadata routing remain absent.

Native tests and eight reference cases cover independent capabilities, nested
and fused-prefix dispatch, cloning, refits, failed fits, and real DummyClassifier
integration for Float32/Float64 inputs. Real-estimator checks include sorted
class columns, output independence, zero probabilities with negative-infinite
logs, and single-class refits. This adopted DummyClassifier returns Float64
probabilities for both input dtypes; Pipeline preserves that type. Probabilities
compare exactly on these fixtures; logarithms use rtol/atol `1e-12` to allow
native math rounding. Negative infinity must match exactly.

## Ordered class labels

`pipe.classes_()` returns the final classifier's ordered labels when the
classifier implements `NumericClassifierClasses`. The pipeline also
implements that trait, so generic consumers can call `get_classes()`, including
through nested pipelines. The result uses the final classifier's target
representation and is owned by the caller. Updating it leaves the fitted
classifier intact. The classifier controls when labels become available.

Mojo exposes `classes_` as a method. Classifiers that expose only a stored
`classes_` field still provide it through `pipe.estimator.classes_`.
The [focused parity check](../../tests/pipeline/classes_parity.py) compares
class order exactly for direct and nested LogisticRegression pipelines,
including clone independence and refits with changed labels. It compares
probability columns within `1e-12` absolute tolerance for solver rounding.

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
Optional-target pipelines also forward `score_samples` through the same trait.
`score_samples` support does not imply `score` support.

Six exact comparisons with local scikit-learn cover scalar dispatch, cloning
and refits, fused prefixes, nesting, and dense Float32/Float64 sample order.
Native contracts additionally cover failed fits, invalid shapes, empty rows,
output ownership, and capability rejection. These are explicit composition
fixtures, not implemented density or outlier algorithms.

## Factories

`make_pipeline(transformer, estimator)` builds the two-stage Pipeline with
sklearn's automatic names. The factories lowercase reflected type names and
append `-1`, `-2`, ... on collision. Estimators backed by a shared struct,
such as `PLSRegression` and `CCA`, can supply their public class name through
`EstimatorClassName`. `make_union(a, b)` and
`make_union(a, b, c)` build a flat FeatureUnion the same way and accept
`n_jobs`, `verbose` and `verbose_feature_names_out`; select Float32 branches
with `make_union[DType.float32](...)`.

```mojo
from moxie.pipeline import make_pipeline, make_union
var pipe = make_pipeline(StandardScaler(), SVC())   # "standardscaler", "svc"
var union = make_union(PCA(n_components=1), StandardScaler(), PCA(n_components=2))
# branches "pca-1", "standardscaler", "pca-2"
```

**Mojo adaptations:** sklearn's variadic `*steps` is a fixed-arity overload
set (two pipeline steps; two or three union branches), because each step has
its own static type. Longer pipelines nest `Pipeline` explicitly, choosing the
inner names. `memory`, `transform_input` and `verbose` are not accepted
because Pipeline does not implement them. Four naming cases compare exactly
with the pinned factories.

## Supervised predictors with transforms

`Pipeline` implements `FitTransformer` when the final predictor implements
`FitTransformer` or `TransformerMixin`. Each step uses its fused method when
available and otherwise fits then transforms. A training-only pipeline can
serve as the final predictor of a nested pipeline and still supports
`predict`. Inference-time `transform` and `TransformingPipeline` require
`TransformerMixin` on the final predictor. A prefix always needs an
inference-time transform. Supervised pipelines require `y` for training.
Omitting it from the shared `FitTransformer` signature raises `ValueError`
before either step is fitted.

```mojo
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler

var pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
var training_projection = pipe.fit_transform(X, y.copy())
var query_projection = pipe.transform(X_query)
var predictions = pipe.predict(X_query)
```

Availability can also depend on the estimator's configuration. LDA's
`lsqr` solver raises on projection. The discriminant analysis parity suite
checks training and held-out projections in both dense floating-point dtypes.
The [training-only parity check](../../tests/pipeline/supervised_training_transform_parity.py)
compares exact scalar outputs for fused dispatch, prediction, refits and nesting
against the pinned scikit-learn pipeline.

## Transformer-only pipelines

`TransformerPipeline(first, last)` is a concrete implementation of the
`TransformingPipeline` capability. `make_pipeline(first, last)` selects
it for two transformers with optional-target `fit`. The pipeline is copyable
when both children are copyable. The first output and last input must match.
The optional target follows the final step's type. A prefix with a different
target type receives `None`. The pipeline prefers each child's `fit_transform`
where a transformed result is needed.

```mojo
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler

var numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
var transformed = numeric.fit_transform(X)
var names = numeric.get_feature_names_out()
```

Use the transformer-only pipeline as a ColumnTransformer branch or nest
two pipelines for more steps. It shares the predictive pipeline's typed
parameter records, clone semantics and feature-name forwarding. The fitted
query follows the final step. `fit` fits that step and `fit_transform`
also returns its result.

When both children implement `InverseTransformer`, the pipeline exposes
`inverse_transform` through that same trait. It applies the last child's
inverse transform first, then the first child's. The
[parity test](../../tests/pipeline/transforming_parity.py) checks Float32
and Float64 output before and after refitting against the pinned
scikit-learn implementation.

## Unsupervised final estimators

`UnsupervisedPipeline(prefix, final)` and `make_pipeline(prefix, final)` compose
an optional-target transformer with an `UnsupervisedEstimator`. The prefix
output and final input must have the same type. The prefix receives the final
estimator's optional target when the types agree and `None` otherwise.
`UnsupervisedPipeline` and `TransformerPipeline` are two names for the same
typed engine.

```mojo
from moxie.cluster import DBSCAN
from moxie.neighbors import KernelDensity
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler

var clusters = make_pipeline(StandardScaler(), DBSCAN(eps=0.5))
var labels = clusters.fit_predict(X)

var density = make_pipeline(StandardScaler(), KernelDensity(bandwidth=0.5))
_ = density.fit(X)
var log_densities = density.score_samples(X_query)
var log_likelihood = density.score(X_query)
```

Training prefers the prefix's `fit_transform` and otherwise calls its
`fit` and `transform`. `fit_predict` delegates through `FitPredictor` and
returns the final estimator's labels directly. Prediction on unseen rows
requires `Predictor`, independently of `FitPredictor`. The pipeline forwards
transformation, per-sample scoring, and decision values when the final
estimator declares those capabilities. `score` follows `UnsupervisedScorer` and
uses the final estimator's aggregation. A concrete final-step method beyond
these traits is accessible on `pipe.estimator`, with input already transformed
by the caller. Configuration-dependent methods retain final-step validation.

`predict_proba` and `predict_log_proba` require the final step's
`ProbabilityPredictor` and `LogProbabilityPredictor` traits, respectively.
For example, `make_pipeline(StandardScaler(), GaussianMixture(n_components=2))`
exposes `predict_proba`. Its columns follow the fitted mixture's component
order and retain the mixture's Float32 or Float64 output dtype. The
[mixture parity check](../../tests/pipeline/mixture_probability_parity.py)
exercises both dtypes, including nested pipelines and refits.

### Training-only transformation

An unsupervised final step can provide `FitTransformer` independently of
`TransformerMixin`. For example, `make_pipeline(StandardScaler(), ClassicalMDS())`
supports `fit` and `fit_transform` on the training data. Its training embedding
keeps the final estimator's Float64 output type for either supported input
dtype. The pipeline exposes `transform` only when the final step implements
`TransformerMixin`, and `inverse_transform` when both steps implement
`InverseTransformer`. A training-only pipeline can be the final step of
another unsupervised pipeline. Steps before the final one must transform
new inputs.

The [embedding parity check](../../tests/pipeline/embedding_parity.py)
compares training, cloning and refitting against pinned scikit-learn.

### Outlier training labels

```mojo
from moxie.neighbors import LocalOutlierFactor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler

var outliers = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=2))
var labels = outliers.fit_predict(X)
```

LocalOutlierFactor uses its training-neighbor scores and returns a rank-one
array of `+1` and `-1` in its configured Float32 or Float64 dtype. With
`novelty=True`, the typed method raises an AttributeError-style error when
the final step runs. This occurs after fitting the prefix. Python checks
availability before fitting either step. OneClassSVM also implements
`FitPredictor` and returns rank-one Int64 labels. Pipeline uses the
`fit_predict(X, y=None)` trait signature. The concrete OneClassSVM method
also accepts sample weights and provides overloads for Float32 input and
custom kernel providers.

The [outlier parity check](../../tests/pipeline/outlier_parity.py) compares
training labels, LOF refits, and nested OneClassSVM query predictions against
the pinned reference. Labels compare exactly. LOF training scores allow
absolute tolerances of `2e-6` for Float32 and `1e-12` for Float64 to account
for rounding in scaling and nearest-neighbor distances.

With the locked Mojo toolchain active, run the focused check from the root:

```sh
python shared/assembly/tools/assemble_family.py --consumer "$PWD" \
  --lock "$PWD/DEPENDENCIES.json" --output "$PWD/builds/deps"
mkdir -p builds/tests/pipeline
mojo build -O3 --fp-mode=contract=off -I builds/deps -I tests/pipeline \
  -I tests/compose/support tests/pipeline/test_pipeline_outliers.mojo \
  -o builds/tests/pipeline/test_pipeline_outliers
python tests/pipeline/outlier_parity.py
```

Use Python with scikit-learn built from the revision recorded in
[interfaces](../interfaces.md).

### Clustering and density validation

The [parity check](../../tests/pipeline/unsupervised_parity.py) compares Float32
DBSCAN labels, Float64 nested KMeans labels and distances, KDE log densities
and their sum, and LocalOutlierFactor novelty predictions and scores against
the pinned reference. Labels and predictions compare exactly. Distances and
scores use an absolute tolerance of `1e-12` for floating-point rounding.
Run the script with Python from an environment where scikit-learn is built
from the pinned checkout, after building `test_pipeline_cluster.mojo` and
`test_pipeline_density.mojo` into `builds/tests/pipeline/`.

## Current boundary

This implementation exposes fit, final-step weighted fit, conditional fit_predict,
predict, conditional transforms and score, per-sample scores, decision and probability methods,
ordered numeric classes, feature names, input feature counts, configuration,
typed step access, read-only slicing, length, cloning and the two factories. The following sklearn Pipeline
API is **deferred** (absent from the typed API, so use fails at compile time):

- `passthrough`/`None` steps and a heterogeneous `named_steps` mapping;
- constructor options `memory`, `verbose` and `transform_input`;
- arbitrary fit parameter routing, metadata routing outside final supervised
  scoring, callbacks,
  `set_output` and `feature_names_in_`.

[Dense FeatureUnion](feature-union.md) is implemented and can be a Pipeline
prefix. Sparse execution remains deferred. This is not full sklearn parity.

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
