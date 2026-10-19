# Serial dense cross-validation

Implemented native APIs: `cross_validate`, `cross_val_score`,
`cross_val_predict`, `validation_curve`, `learning_curve`,
and `permutation_test_score`, using the family `TypedScorer`, `FitAdapter`,
`ParameterAdapter`, `ScoreTable`, `FitStatus`, `ErrorPolicy` and
foundation `clone` contracts.

## Explicit typed interface

Calls specialize on an adapter type, e.g. `cross_validate[MyAdapter](...)`.
`CVAdapter` extends FitAdapter with `n_samples`, `input_rows`, `target_rows` and
`metadata_rows`. Input slicing receives both selected rows and training rows so
an adapter can explicitly implement pairwise-kernel indexing. Metadata slicing
must separate sample weights from unsliced options. Associated inputs, targets,
metadata and scorers must be deinitializable; the engine states these constraints
without changing the inherited foundation or family fit/scorer traits.

Pass the estimator, X, y, adapter, a list of typed scorers, matching metric names,
a materialized `List[IndexSplit]`, fit metadata and score metadata. The splitter
cursor supplies the pairs; `check_cv` remains a separate explicit selection step.
No implicit classifier detection, dynamic scorer lookup or metadata routing occurs.
All scores follow the supplied scorer's higher-is-better convention.

`cross_validate` fits one independent unfitted clone per fold, then calls every
scorer on that fitted clone. It returns `CVResult`: a ScoreTable with candidate
axis length one, a status record per fold, and optional estimator/index lists.
`return_train_score`, `return_estimator` and `return_indices` default to False.
Absent options remain None. Timings use a monotonic clock; fit_time measures
adapter.fit itself (excluding row materialization), and score_time measures test
scoring only. Failed fits have zero score_time.

Default `error_score` is `ErrorPolicy(False, NaN)`. For raising behavior pass
`ErrorPolicy(True, 0)`; the numeric field is then ignored. Fit failures fill all
metrics, individual scorer failures fill only that scorer, and diagnostics are
owned strings. All-fit failure raises. The private search entry point can defer this check
until all candidates have been evaluated; the public CV behavior is unchanged. Warning emission is not implemented.
Original estimators stay unchanged, including when already fitted.

`cross_val_score` accepts one typed scorer and returns a Float64 vector.
`cross_val_predict` fits independent clones, validates that held-out indices
partition all rows exactly once before fitting, and restores predictions to
original row order. Numeric, String and class-response adapters are described
below. Fit/prediction errors always propagate.

Execution is serial. Python parallelism, `pre_dispatch`, routing and scorer-name
reflection are not exposed. Sparse inputs, slicing and predictions remain deferred.
Incremental adapters and native SVG displays are implemented as described below.

## Out-of-fold predictions

`cross_val_predict[A]` accepts an estimator, X, y, adapter, explicit split list
and fit metadata. Choose one prediction-adapter capability per adapter type so
the result type is unambiguous at compile time:

- Existing `PredictionAdapter`: rank-one/two numeric `NDArray[PredictionDType]`,
  unchanged return type. The adapter's `predict` method selects the native
  estimator operation; the optional public `method` accepts only `"predict"`.
- `StringPredictionAdapter`: `predict` returns the shared `StringTarget`; the
  engine preserves its rank and number of output columns. This also accepts
  only `method="predict"`, and returns an owned `StringTarget`.
- `ClassPredictionAdapter`: requires explicit `method="predict_proba"`,
  `"predict_log_proba"` or `"decision_function"`. Associated `ResponseDType` is
  Float32 or Float64. Unsupported estimator methods must raise in the adapter;
  implementing probability prediction does not require an actual decision method.

All three use `CVAdapter` for feature/target/metadata slicing. Train-square and
test-vs-train pairwise slicing stays explicit. Scorers supply associated types
but are not called. Each fold gets a fresh unfitted clone, including when the
original is fitted. Metadata stays aligned with selected original sample rows.
Rank, row count and output width must remain consistent across folds. No caller
arrays, splits, targets or estimator state are mutated.

Class-response adapters provide three methods:

1. `encode_target(y) -> IndexArray`: encode the full target into sorted,
   contiguous global class codes, preserving rank. Use the supplied
   `encode_prediction_target` overloads for Int64/Float32/Float64 arrays or
   `StringTarget`; these call the shared label owner independently per column.
2. `fit_encoded(mut estimator, X, encoded_y, metadata)`: fit on those global
   codes. Do not renumber the classes within each fold.
3. `response(estimator, X, method) -> ClassResponse[dtype]`: return owned
   `values: List[NDArray[dtype]]`, matching `classes: List[IndexArray]` of fitted
   global codes, and `is_list`. Single-array predictions have exactly one entry
   and `is_list=False`. Multioutput-multiclass predictions use one entry per
   target column with `is_list=True`. A single response matrix for indicator
   targets instead uses output-column IDs as `classes`, as in the reference.

The engine checks sorted, unique, in-range class IDs and aligns missing columns.
Missing `predict_proba` columns receive zero; missing log-probability and
multiclass decision columns receive the response dtype's **smallest finite
value**, not negative infinity. Existing negative infinities in log predictions
are preserved. A decision fold missing classes with at most two fitted classes
raises. Missing-class irregular decision matrices (such as OVO pairwise columns)
also raise. Full-class binary decision vectors and full-class OVO matrices retain
their original shape. No Python missing-class warnings are emitted.

Class methods return `CVClassPrediction[dtype]`, a native adaptation of Python's
array-or-list return: `values` contains restored output arrays and `is_list`
preserves which form the estimator produced. `array()` returns an owned snapshot
for a single-array result and raises for list results. This wrapper avoids
heterogeneous dynamic return types; numeric and String predictions keep their
direct return types. The [runnable example](../examples/cross_val_predict.mojo)
shows a complete DummyClassifier adapter and rare-class probability alignment:

```mojo
var result = cross_val_predict[ClassifierAdapter](
    DummyClassifier(), X, y, ClassifierAdapter(), splits, None,
    method="predict_proba",
)
var probabilities = result.array()
```

The normal typed-CV scope still applies: no sparse assembly, automatic
classifier/scorer discovery, Python metadata routing or dynamic reflection.
For target-less numeric operations, the existing `PredictionAdapter` can select
the operation in its own `predict` implementation; no class alignment is inferred.

## Validation curves

`validation_curve[A]` accepts an adapter implementing both `CVAdapter` and
`ParameterAdapter`. The latter owns `Value` and
`configure(mut estimator, Dict[String, Value])`: apply partial constructor
parameters, reject unknown names, and clone nested estimator configurations.
`SearchAdapter` already includes this capability, so the same adapter works
with curves and grid/random search. Neither the curve engine nor this capability
requires prediction, classifier detection, or a concrete estimator import.

Pass one typed `scoring` value, explicit materialized `cv` splits, fit metadata
and score metadata, plus keyword-only `param_name` and `param_range`. Like the
other native CV functions, implicit `cv=None` / `scoring=None`, Python scorer
lookup and metadata routing are not exposed. Use `check_cv` or a splitter to
choose folds; groups enter that selection, not curve fitting.

The result is `(train_scores, test_scores)`, two owned Float64 arrays shaped
`(len(param_range), len(cv))`. Values retain their input order, including
duplicates. Each fold evaluates all values before advancing to the next fold,
using one fresh unfitted clone per value. The original estimator, input arrays,
parameter list and split indices are borrowed and remain unchanged. There is
no ranking, best-parameter selection or full-data refit. Storage uses the existing
ScoreTable internally and materializes the two output arrays with NuMojo reshape.

Numeric `error_score` replaces both scores for failed fits; a scoring failure
replaces only the affected train or test score. Even if every fit fails, the
curve returns replacement scores, matching the pinned reference. This differs
from `cross_validate`'s aggregate all-fit-failure error. Default replacement is
NaN; `ErrorPolicy(True, 0)` propagates fit/scoring errors. Configuration, slicing
and invalid-index errors always propagate. Empty parameter ranges and empty
split lists raise a descriptive ValueError-prefixed native Error; Python's
incidental empty-result aggregation exception is not reproduced. Curves return
only scores; warning delivery and diagnostic result extensions are not exposed.

The runnable [example](../examples/validation_curve.mojo) compares a real
DummyRegressor's mean and median strategies:

```mojo
var scores = validation_curve[RegressionAdapter](
    DummyRegressor(), X, y, RegressionAdapter(), R2Scorer(), splits,
    None, None, param_name="strategy", param_range=strategies,
)
# scores[0]: train scores; scores[1]: test scores, indexed by strategy then fold.
```

Run `pixi run test` here to compile and execute the complete example, including
its explicit adapter and fold construction.

## Learning curves

`learning_curve[A, size_dtype=DType.float64]` needs only `CVAdapter`. Pass the
estimator, X, y, adapter, one typed scorer, explicit `cv` splits, fit metadata
and score metadata, as with validation curves. Prediction and parameter-setting
capabilities are not required. Targets may be optional or multioutput when the
adapter supports them; pairwise adapters receive the actual training prefix for
both square training kernels and test-vs-training rectangles.

The keyword `train_sizes` accepts an optional NuMojo vector. With the default
`size_dtype=DType.float64`, values are fractions in `(0, 1]`; specialize with
`DType.int64` for absolute counts. Omitting the vector (native `None`) uses the
five reference fractions from `linspace(0.1, 1.0, 5)`. Fractions are multiplied
by the **first** fold's training length, truncated to integers and clipped to
at least one. Both forms are sorted and deduplicated. Counts must be between
one and that first training length. Later shorter folds use their available
prefix; the returned nominal sizes still refer to the first fold. Longer folds
do not increase the requested sizes.

The result is a `LearningCurveResult`, a typed native adaptation of Python's
variable-length tuple:

- `train_sizes`: owned Int64 vector of sorted unique absolute sizes.
- `train_scores`, `test_scores`: owned Float64 matrices `(size, split)`.
- `fit_times`, `score_times`: optional Float64 matrices of the same shape,
  present only with `return_times=True` (default False).

Evaluation proceeds split-first, then size-first. Every evaluation clones the
original estimator into an unfitted instance; fitted originals and all caller
arrays/splits remain unchanged. Each test fold stays fixed while training uses
nested prefixes. Sample-aligned fit and score metadata follow those exact rows;
unsliced options remain the adapter's responsibility. Timings follow the shared
CV engine: fit time measures adapter fitting, excluding input/metadata
materialization; score time measures held-out scoring, excluding train scoring.
Failed fits have zero score time. Durations depend on the machine and are not
expected to numerically match Python's timings (which also include input slicing).

`shuffle=False` is the default and consumes no RNG draws. With `shuffle=True`,
an integer `random_state` reconstructs a stream each call. Alternatively, pass a
`mut RandomState` immediately after score metadata; that stream advances across
calls. Each entire training fold is permuted once, before any fits, and all sizes
use prefixes of that permutation. Caller splits are never shuffled in place.
The adopted RNG snapshot has no shared default stream: `shuffle=True` without
an integer seed or explicit mutable stream raises, consistent with this family's
existing splitter adaptation. No local RNG or replacement global stream is added.

Default `error_score` is `ErrorPolicy(False, NaN)`. Failed fits replace both
scores; individual scoring failures replace only the failing side. Unlike
`validation_curve`, **all fits failing raises even with numeric error_score**.
`ErrorPolicy(True, 0)` propagates the first fit or score error. Invalid indices,
slicing errors, malformed/empty size vectors and empty split lists propagate.
A first fold with no training rows and nonfinite fractional sizes explicitly
raise; Python's incidental empty/NaN conversion behavior is not reproduced.
Duplicate removal and partial failures do not emit Python warnings.

`exploit_incremental_learning=True` requires `IncrementalCVAdapter`. It clones
once per fold, passes only each new training chunk to partial_fit, and scores
the accumulated prefix. Adapters receive the full target to establish classes.
Partial-fit failures propagate; numeric error_score applies to scoring failures.
Sparse data, implicit CV/scorer discovery, parallelism, Python routing and
verbosity remain unsupported.

The runnable [learning-curve example](../examples/learning_curve.mojo) reuses the
explicit dense regression adapter from the validation-curve example:

```mojo
var result = learning_curve[RegressionAdapter, DType.int64](
    DummyRegressor(), X, y, RegressionAdapter(), R2Scorer(), splits,
    None, None, train_sizes=sizes.copy(), shuffle=True, random_state=42,
    return_times=True,
)
# result.train_scores[size, fold], result.test_scores[size, fold]
```

## Permutation test scores

`permutation_test_score[A]` accepts the same typed adapter/scorer, explicit split
list and separate fit/score metadata as the other CV functions. Defaults are
`n_permutations=100` and integer `random_state=0`. An overload accepts a mutable
`RandomState` after score metadata. A process-global `None` seed is not exposed
by the adopted RNG snapshot.

The returned `PermutationTestResult` contains `score` (the unweighted mean fold
score), an owned Float64 `permutation_scores` vector, and `pvalue`. The latter
is `(count(permutation_score >= score) + 1) / (n_permutations + 1)`. Ties count;
NaN comparisons remain false as in the reference. Fit/scoring errors propagate;
there is no `error_score` replacement policy. At least one permutation is required.

Only target rows are shuffled, always starting from the original targets.
Features, groups and fit/score metadata stay aligned to the original sample
positions. A rank-two target is permuted by whole rows. Optional `groups` takes
`SplitLabels`, preserving sorted numeric or String group order through the shared
label encoder. Each group is independently permuted in original row order;
singleton groups consume no RNG draws. Groups must be rank one and sample-aligned.

A materialized split list has the same semantics as sklearn's fixed iterable CV.
For target-dependent splits such as stratification, implement `PermutationCV`
with associated `Adapter` and `split(mut self, X, y, groups) -> List[IndexSplit]`.
Call `permutation_test_score[YourCV]` with that mutable provider in place of the
list. It is called once for the original targets and once per shuffled target;
its state persists. This deliberately does not freeze stratified folds computed
from the original labels. `FixedPermutationCV[A]` is the list wrapper used by
the convenience overload. Group labels reach the provider unchanged.

The pinned reference clones once per dataset, then refits that clone across its
folds. This is intentionally different from `cross_val_score` and matters for
warm-start estimators. Original fitted/unfitted estimator state stays untouched;
each permutation begins with a new clone. An invalid split list raises before
fitting that dataset. Metadata row selection and pairwise inputs use `CVAdapter`.

See the runnable [permutation example](../examples/permutation_test_score.mojo).
The algorithm follows Test 1 in [Ojala and Garriga (2010)](https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf).
Implicit CV/scorer lookup, Python routing/warnings, parallelism and sparse inputs
remain outside the native interface.

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity` from this child.
Curve consumers use `-O3`, as these tasks do: on the pinned Mojo 1.0 / NuMojo
combination, reshape fails LLVM masked-gather lowering at `-O0`. The family
public-import curve probe also uses `-O3`; numerical parity disables FP
contraction with `--fp-mode=contract=off`.
Native probes check clone isolation, one fit for multiple metrics, already-fitted
originals, partial/all fit failures, score errors, optional results, weights,
partition rejection, and Float32 matrix prediction restoration. The actual implemented
DummyRegressor, read live from Moxie core (`moxie.dummy`), supplies end-to-end tests.
It is a test dependency only; the generic engine imports no estimator.

Parity executes the original sklearn CV function bodies at revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b with sklearn 1.9.1 infrastructure and
minimal NumPy device adapters, plus the pinned DummyRegressor class. The default
reference interpreter is Moxie's Pixi Python under the Moxie root `.pixi` directory; override
`MOXIE_REFERENCE_PYTHON` with another interpreter providing sklearn 1.9.1.
Python remains oracle-only. The initial fixture compares four fold R2 values and
eight predictions at rtol/atol 1e-13, allowing Float64 reduction rounding; this
small fixture does not establish broad sklearn estimator or classification parity.
Curve parity adds real mean/median strategy sweeps with duplicate values,
fit-only and fit/score weights, numeric/default-NaN all-fit failures and scorer
failures. Float64 score comparisons use rtol=atol=1e-12 for bounded fixtures to
allow reduction rounding. Capability-only probes verify fresh clones,
split-first execution order and per-fold/per-side failures exactly; they are
not implemented estimators. Native tests also check invalid parameters, empty
ranges/splits, bounds, repeat calls and fitted/unfitted original isolation.
Search tests compare both APIs' per-fold scores using one unchanged SearchAdapter.
Learning-curve parity adds default, fractional and integer sizes, duplicates,
unequal fold lengths, strided size input, integer/mutable RNG streams, fit-only
and fit/score weights, and fit/scorer failures. Sizes, probe execution order and
the subsequent RNG word match exactly; real DummyRegressor scores use
rtol=atol=1e-12 for bounded Float64 reduction rounding, with NaNs compared as NaNs.
Native checks also cover optional timing shapes/nonnegative values/failed-fit
zeros, input and original-estimator isolation, invalid sizes and indices, and
explicit incremental-mode rejection. Capability probes exercise a CV-only adapter
with Float32 pairwise kernels and optional/multioutput targets; these probes are
not implemented estimators. These checks do not establish arbitrary-estimator
or sparse parity.
Permutation parity compares real DummyRegressor observed/null scores, exact
p-values and subsequent RNG state for unrestricted, String-grouped, singleton,
weighted and dynamically stratified cases. Lifecycle probes check one clone per
dataset, fold refits, ties, NaNs and early index failures. Score tolerances are
rtol=atol=1e-12 for bounded Float64 reduction rounding.
Prediction parity adds real DummyClassifier String predictions, weighted
probabilities/log-probabilities, multioutput class lists and missing-class binary
folds. Capability-only probes compare Float32/Float64 multiclass decisions,
binary vectors, full-class OVO matrices, multilabel indicator responses and
reference decision errors. Class identities, missing-column sentinels and probe
responses compare exactly; Float64 probability/log values allow rtol=atol=1e-12.
Native tests also cover large Int64 labels, malformed class IDs, unsupported
methods and original estimator isolation. These are not claims of arbitrary
classifier or full sklearn parity.
No performance claim is made.

## Native displays and incremental validation

LearningCurveDisplay and ValidationCurveDisplay copy input data, expose typed
mean/population-standard-deviation artist records and write SVG through
`figure_.value().savefig(path)`. Plotting supports train/test/both, score
negation, fill/errorbar/no spread, finite x values and automatic linear/log/symlog
scales. At least two numeric x values and nonempty folds are required. These
are native records, not matplotlib objects; arbitrary Python plotting kwargs
and axes are unsupported. `plot` mutates the display and returns None.

Incremental GaussianNB checks use current Moxie Naive Bayes with pinned local
sklearn function/class bodies for 36 score cells and exact RNG continuation.
Native lifecycle tests cover incremental metadata slicing, clone isolation,
partial-fit propagation and scoring failures. Display tests check copied data,
styles, 12 mean/std values against NumPy's reference formulas, XML escaping
and parsed SVG output. These bounded checks do not establish full plotting
or arbitrary incremental-estimator parity.
