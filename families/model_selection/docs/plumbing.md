# Model selection plumbing

This family supplies contracts and validation/ownership helpers. Concrete
splitters are owned by `algorithms/splitters` and exported through the verified
assembly. The validation child supplies serial CV engines; the search child supplies
candidate enumeration and serial grid/random search. Threshold estimators remain queued. Existing `CrossValidator`, `SearchEstimator` and
`PredictiveSearch` remain unchanged. Python module boundaries guide future
`_split`, `_validation`, `_search`, `_classification_threshold`, and
`_search_successive_halving` contributions. Private contract modules are a
native adaptation that keeps evaluation independent of concrete estimators.

## Shared interfaces and layouts

| Owner/module | Interface | Contract |
| --- | --- | --- |
| scaffold / `_indices` | `IndexArray = NDArray[int64]`; `IndexSplit(train, test)` | Rank-one owned indices, original row coordinates. `copy()` deep-copies both buffers; `validate(n_samples)` checks rank and bounds. No implicit sorting or deduplication. |
| scaffold / `_indices` | `SplitCursor.next(mut self) raises -> Optional[IndexSplit]` | Stream owns each yielded pair; `None` ends iteration. Fresh cursors implement repeated traversal. A splitter can use this as `CrossValidator.Splits`; the inherited associated type stays extensible. |
| scaffold / `_evaluation` | `TypedScorer` | Associated `Estimator: BaseEstimator & Deinitable`, `Input`, `Target`, `Metadata`; `score(self, estimator, X, y, metadata) raises -> Float64`. Fitted estimator and arrays are borrowed. Higher is better; loss adapters negate once. |
| scaffold / `_evaluation` | `ParameterAdapter: FitAdapter` | Associated copyable `Value`; `configure(mut estimator, Dict[String, Value])` applies partial constructor parameters, rejects unknown names and clones nested estimator values. Shared by search and validation curves. |
| scaffold / `_evaluation` | `FitAdapter` | Associated `Scorer: TypedScorer`; `fit(self, mut estimator, X, y, metadata) raises`. Types are exactly the scorer's associated types. Provides explicit supervised, unsupervised, weighted or other capability dispatch without imposing prediction or probability. |
| scaffold / `_evaluation` | `clone_fit_score[A: FitAdapter]` | Arguments: estimator, adapter, scorer, X_train, y_train, X_test, y_test, fit_metadata, score_metadata. Calls foundation clone, then adapter fit, then scorer score. Call with explicit adapter type, e.g. `clone_fit_score[MyAdapter](...)`, because associated-type inference is not automatic on the pinned compiler. Returns Float64; every error propagates; original estimator stays borrowed. |
| scaffold / `_results` | `ScoreTable(test_scores, train_scores, metric_names)` | Float64 NuMojo rank-three `(candidate, split, metric)`. Train scores optional with identical shape. Names are unique nonempty owned strings in metric order. `validate()` checks these invariants. |
| scaffold / `_results` | `FitStatus(failed, message, fit_time, score_time)` | One per candidate/split, candidate-major order. Times Float64 seconds; diagnostics owned strings. Status is separate from NaN scores. |
| scaffold / `_results` | `ErrorPolicy(raise_on_error, error_score)` | Explicit raising/numeric policy. Child evaluation handles warnings/all-fits-failed; final refit always raises. No suppression in the lifecycle helper. |
| scaffold / `_results` | `RefitSelector.Results`; `select(self, results) raises -> Int` | Typed callback receives the full child result record, not only scores. Validate index using `validate_refit_index(index, n_candidates)`. |
| scaffold / `_results` | `require_refit(enabled, fitted)` | Clear error for refit=False or missing learned estimator before delegated inference. Compile-time wrapped capability is independently required. |
| scaffold / `_candidates` | `Candidate[Params: Copyable & Deinitable](params, ordinal)` | Constructor configuration, no learned state; nonnegative stable ordinal. Params must implement deep configuration ownership for nested estimators. |
| scaffold / `_candidates` | `ParameterCandidates.Params`; `next(mut self) raises -> Optional[Params]` | Independent configurations in deterministic enumeration order; no generation algorithm here. |
| scaffold / `_candidates` | `ParameterDistribution.Value`; `sample(self, mut random_state: RandomState) raises -> Value` | Typed distribution extension, using the single shared RNG. |

Numerical arrays use NuMojo directly. Feature arrays are Float32/Float64 rank-two
`(sample, feature)`; target arrays retain rank-one or multioutput rank-two shape
and estimator-specific label dtype. Scorer associated types permit nonnumeric
label records, optional targets and explicit metadata records. Metadata must
separate sample-aligned weights from unsliced fit options. Pairwise estimators
need an explicit adapter that slices training square kernels and test-vs-train
rectangles. These are child responsibilities, not implicit generic row slicing.

Inputs are borrowed and returned arrays move. `.copy()` owns independent data;
`.view()` shares mutation. Element strides and offsets follow the numerical
owner, including negative strides; copying a strided view can copy its entire
backing buffer. Do not retain caller aliases as fitted state. Arrays of score
columns use Float64 regardless of estimator dtype. Lists hold ragged records,
strings, configuration and diagnostics, not replacement dense numeric storage.

Split validation intentionally permits empty arrays, duplicates and overlap:
custom user CV iterables can provide these, and split algorithms enforce their
own stronger rules. `cross_val_predict` must separately verify that test indices
partition all rows exactly once. Ordinary CV and repeated splits must not apply
that partition restriction. Training/test row order must be retained.

## Child result and lifecycle obligations

Validation owns its result record: `scores: ScoreTable` with candidate dimension
one; `status: List[FitStatus]`; optional fitted estimators and optional owned
`List[IndexSplit]` when requested. It exposes sklearn's named score/time/indices
fields through typed accessors. Absent options are `None`, never dummy arrays.
`validation_curve` exposes two Float64 `(parameter, split)` arrays and retains
split-first execution order; numeric failures do not trigger CV's aggregate
all-fits-failed guard. `learning_curve` returns a `LearningCurveResult` with an
owned Int64 training-size vector, Float64 train/test score matrices `(size, split)` and optional fit/score
time matrices. It sorts/deduplicates sizes against the first training fold,
fits fresh clones in split-first order, and raises when all fits fail.
Prediction results restore original row order; class/probability columns follow
the full target class order with explicit missing-class behavior. Numeric
`PredictionAdapter` and `StringPredictionAdapter` return direct arrays/targets;
`ClassPredictionAdapter` uses `ClassResponse` for fitted class codes and returns
`CVClassPrediction` to preserve the single-array versus multioutput-list form.
Global target encoding delegates to the shared label owner before fold fitting.
Curves add train-size or parameter axes before split/metric axes; permutation
results hold
observed score, Float64 permutation vector and p-value. Displays consume these
native data records; Python plotting objects are separate binding work.

Search owns `cv_results_`: ordered `List[Candidate[Params]]`, raw `ScoreTable`,
status records, Float64 mean/std columns `(candidate, metric)`, Int64 rank
columns of that shape, timing columns `(candidate,)`, and typed optional
parameter fields (the native equivalent of masked heterogeneous dictionaries).
Candidates and splits retain enumeration order. Population standard deviation,
unweighted split means, minimum ranks for ties, first candidate on tied rank,
and reference NaN handling belong to its tested aggregation implementation.
Callable refit may access all fields, selects a checked index and omits
`best_score_`. Single-metric refit=False still exposes best index/params/score;
multimetric refit=False does not invent a best candidate. Only successful refit
creates `best_estimator_`; refit trains a fresh selected configuration on all
input data. Failed or repeated fits must not expose stale learned results.
Halving extends results with iteration/resource columns; thresholds preserve
binary class order, response method and positive-label semantics.

Ordinary CV and curves clone before each independent fit. The reference
permutation-test lifecycle instead clones once per original/permuted target and
refits that instance across its folds, including observable warm-start behavior.
Integer seeds reconstruct streams
at reference check_random_state boundaries. Mutable streams advance by explicit
borrow, and RNG context ownership follows foundation.random. Nested estimator
parameters must clone recursively. Return-estimator mode retains each fitted
clone; the scalar helper deliberately discards it, so engines requiring that
option compose the same clone/adapter/scorer calls themselves. Multi-metric
scoring performs one fit and multiple scorer calls, not one refit per metric.

Optional capability constraints remain separate. An unsupervised adapter can
use `Target = Optional[T]`; weights enter only supported typed fit/score
metadata. Scorers that need predict_proba or decision_function require those
traits on their concrete estimator type. A CV-enabled estimator consumes this
plumbing without importing validation's concrete algorithms; validation depends
only on capabilities, which avoids cycles.

## Ownership and dependency map

Each row names the architectural owner; the splitter child now contains concrete
implementations. Validation, validation-curve, learning-curve, candidate and
grid/random-search and permutation-test engines are implemented; thresholds,
halving remain queued.

| Child | Public exports | Dependencies |
| --- | --- | --- |
| `algorithms/splitters/` | BaseCrossValidator, BaseShuffleSplit, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, PredefinedSplit, check_cv, train_test_split | model_selection.scaffold |
| `algorithms/validation/` | cross_validate, cross_val_score, cross_val_predict, learning_curve, validation_curve, permutation_test_score, LearningCurveDisplay, ValidationCurveDisplay | scaffold, splitters, dummy.estimators |
| `algorithms/search/` | ParameterGrid, ParameterSampler, GridSearchCV, RandomizedSearchCV | scaffold, validation |
| `algorithms/thresholds/` | FixedThresholdClassifier, TunedThresholdClassifierCV | scaffold, validation, naive_bayes.gaussian |
| `algorithms/halving/` | HalvingGridSearchCV, HalvingRandomSearchCV | scaffold, search |

BaseSearchCV, GroupsConsumerMixin and BaseSuccessiveHalving are private reference
implementation classes. Halving is experimentally enabled in Python; a native
child must document explicit opt-in rather than silently claiming default
Python export availability.

## Assembly and native compatibility

`DEPENDENCIES.json` (assembly schema 1) declares live Moxie, numerics, NuMojo,
RNG, labels, mojo-linalg, the shared assembly tool and this scaffold.
`EXPORTS.json` names package exports explicitly. `tools/check.py` invokes the
current `../../../shared/assembly` launcher, which reads every declared source
from its live checkout on each build. There are no revision/hash pins, snapshots
or local assembler; missing sources fail and preserve existing output.

Children add their owned module/export maps and all transitive dependency
declarations, including this family's contracts, to their own manifest. Rebase
relative project paths when creating that manifest. Compile against the one generated
`builds/deps` namespace; never stack partial moxie import roots. Identical
baseline modules use the foundation's baseline markers for deduplication.
Different owners of a module or export are rejected. Relocating an existing
algorithm requires the assembler's explicit ownership override and provenance;
shared helper forks are prohibited. Child source remains in its own src/.

`tests/consumers/a` and `b` are isolated fixtures, not future child projects.
Both assemble two distinct estimator contributions plus these exact shared
modules, compile and execute generic lifecycle, dense norms and seeded RNG.
Their outputs and caches are contained, and separate copies are hash-identical.
Tests also reject missing live sources, conflicting module owners and conflicting
exports. The toolchain is Mojo 1.0.0 / MAX 26.5.0 (Pixi-locked) with live NuMojo.
Manifest changes require both consumer probes again.

Run `pixi run build` and `pixi run test`. Existing foundation/family conformance
checks stay active. These checks establish contracts, not algorithm parity.

## Integrated owner

Source and algorithm children are owned by `moxie/families/model_selection`.
Run the focused commands in [README](../README.md) from Moxie.
`IncrementalCVAdapter` opts into learning-curve partial fitting with full-target
class information; ordinary adapters reject incremental mode. LearningCurveDisplay
and ValidationCurveDisplay own copied data and emit native SVG. They require
at least two x values, numeric validation-curve x coordinates and nonempty folds;
Python axes, arbitrary plotting kwargs and metadata routing remain unsupported.
Display plot mutates in place and returns None. Threshold tuning and successive
halving remain explicit raising declarations.
