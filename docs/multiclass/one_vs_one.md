# OneVsOneClassifier

`moxie.multiclass.OneVsOneClassifier` ports `sklearn.multiclass.OneVsOneClassifier`
for dense, single-label targets. It fits one clone of a binary classifier or regressor per
pair of classes, `K * (K - 1) / 2` in all, and predicts the class with the most
pairwise votes. Ties are broken by the summed component confidences. Parity is
checked against the pinned scikit-learn revision in [family.md](family.md).

```mojo
from moxie.multiclass import OneVsOneClassifier

var model = OneVsOneClassifier(MyBinaryClassifier(), n_jobs=None)
model.fit(X, y)                          # X: NDArray[float64] (n, p), y: NDArray[int64] (n,)
var labels = model.predict(X)            # NDArray[int64], values from classes_
var scores = model.decision_function(X)  # (n, K), or (n,) for two classes
```

See [examples/multiclass_one_vs_one.mojo](../../examples/multiclass_one_vs_one.mojo) for a runnable program.

## Type parameters and the component contract

`OneVsOneClassifier[E, dtype=DType.float64, label=DType.int64]`:

| Parameter | Meaning |
| --- | --- |
| `E` | Binary component: `SupervisedEstimator & Predictor & Copyable`, conforming to `RegressorMixin`, or to `ClassifierMixin` plus `DecisionFunction` and/or `ProbabilityPredictor` |
| `dtype` | Feature dtype; `E.Input` must be `NDArray[dtype]` |
| `label` | Class label dtype: `int64`, `float32` or `float64` |

`E.Target` must be `NDArray[DType.int64]`. Classifier predictions must be
`NDArray[int64]`; regressor predictions must be `NDArray[float32/float64]`. For classes
`i < j`, the component is trained only on the samples labelled `classes_[i]`
(target 0) or `classes_[j]` (target 1), in their original order. At prediction
time its `predict` output casts the vote. Classifier confidence comes from
`decision_function` when available, otherwise `predict_proba(X)[:, 1]`.
Decision values must be `NDArray[float32/float64]` of shape `(n,)` or `(n, 1)`,
and probabilities `(n, 2)`. Unsupported types fail to compile; invalid shapes raise.

Regressor predictions have shape `(n,)`. Regressors supply confidence through
`predict`, even when they provide a `decision_function`. Only predictions exactly equal to 0 or 1 cast a vote;
fractional or out-of-range values contribute confidence without voting. They
are neither rounded nor thresholded before voting. The wrapper always exposes
`decision_function` and keeps classification accuracy as its score.

## Behavior

| Item | Behavior (matches sklearn unless noted) |
| --- | --- |
| `classes_` | Sorted unique labels, `NDArray[label]` |
| `estimators_` | `List[E]`, one per pair `(0,1), (0,2), …, (K-2,K-1)` |
| `n_classes_()` | `len(classes_)` |
| `n_features_in_` | `Optional[Int]`, set by every fit |
| `pairwise_indices_` | `Optional[List[NDArray[int64]]]`: original training rows for each pair after pairwise batch fit; otherwise `None` |
| `decision_function` | `votes + s / (3 (abs(s) + 1))` per class, where `s` is the class's summed signed confidence. `(n, K)`, or column 1 as `(n,)` for two classes. Always available |
| `predict` | First argmax of `decision_function` (a NaN wins, as with `np.argmax`). Two classes: strict threshold 0.0 for classifiers with `decision_function`, else 0.5 (including all regressors) |
| `predict_proba` | Not provided, as in sklearn |
| `score` | Mean accuracy, optionally weighted |

Fit validates like sklearn's `validate_data` and raises the same `ValueError`
messages for these cases:
- `X` that is not 2-D, or has no samples or no features
- `y` that is not 1-D or an `(n, 1)` column (the column is flattened)
- NaN or infinite float labels
- continuous targets
- mismatched lengths
- a single class
- `n_jobs=0`

NaN and infinite features pass through to the component, as in sklearn. Predict
rechecks `X` and its feature count. Learned state is committed only after every
component fits, so a failed fit leaves the previous model intact (sklearn may
leave partial state).

## Dense precomputed kernels

A component opts into sklearn's `input_tags.pairwise` semantics by conforming
to the family-local `moxie.multiclass.PairwiseInput` trait and implementing
`pairwise_input(self) -> Bool`. The value describes its current constructor
configuration (for example, whether `kernel` is `"precomputed"`), so it can
change through `set_params`. Undeclared components default to `False` through
the exported `pairwise_input(estimator)` helper. This is an explicit Mojo
capability adaptation, not an implementation of sklearn's full tag system.
OvO and OvR expose the trait and delegate to their template estimator.

```mojo
# MyKernelClassifier implements PairwiseInput and returns True.
var model = OneVsOneClassifier(MyKernelClassifier(), n_jobs=2)
model.fit(K_train, y)                  # square (n_train, n_train)
var predicted = model.predict(K_test) # rectangular (n_test, n_train)
```

Batch fit requires a square matrix. For each sorted class pair it trains on
`K_train[indices, indices]` using the Cartesian product of the selected rows
and columns, preserving sample order. `pairwise_indices_` retains those row
indices in `estimators_` order. `n_features_in_` is the full training width;
each component receives only its pair's width. Prediction first checks the
full width, then sends `K_test[:, indices]` to both `predict` and the confidence
method. NaN/inf, symmetry and positive definiteness checks remain the
component's responsibility, as in sklearn.

These gathers use existing NuMojo arrays and materialize owned subsets.
Training uses two axis gathers with an intermediate containing selected rows
and all columns. Only the indices are retained by the wrapper. Dense Float32
and Float64 arrays, including strided inputs, follow the same contract.
Refitting replaces the indices, non-pairwise fitting clears them, and cloning
starts without them. Inference uses the saved indices even if the template's
configuration has subsequently changed. Parallel jobs keep pair order and
commit components and indices together after all fits succeed.

## Incremental fitting

`partial_fit(X, y, classes=None)` is available only when `E` conforms to
`IncrementalClassifier`, with `E.Classes = NDArray[int64]`. The wrapper also
conforms to `IncrementalClassifier` in that case. Declare all possible classes
on the first call using a rank-one `NDArray[label]`:

```mojo
var model = OneVsOneClassifier(MyOnlineClassifier(), n_jobs=2)
model.partial_fit(X_first, y_first, all_classes)
model.partial_fit(X_next, y_next)
```

Declarations are sorted and deduplicated. Later declarations, if supplied,
must describe the same set; every batch label must belong to that set. Targets
may be a vector or an `(n, 1)` column. The wrapper validates nonempty dense
features, matching sample counts, discrete finite labels and unchanged feature
width before dispatching updates. NaN/inf features reach the component.

All class-pair components are cloned on the first successful call. Each
nonempty pair receives its rows in batch order, binary targets 0/1 and
`classes=[0, 1]` on every update, including batches containing only one member
of that pair. Pairs receiving no rows keep their state, including unfitted
clones. `estimators_` therefore may contain unfitted components even though
the wrapper has learned `classes_` and `n_features_in_`. Both `predict` and
`decision_function` visit all pairs and propagate component inference errors;
there is no substitute vote for a pair that has never received samples.

`partial_fit` after ordinary `fit` continues those fitted components and may
omit `classes`. Ordinary `fit` after incremental updates replaces the learned
state. Changing the template estimator affects subsequent `fit` calls, not
existing incremental components. Cloning the wrapper returns an unfitted model.

Pairwise `partial_fit` follows a different sklearn path: it selects rows only,
passes every input column to each component, and sets `pairwise_indices_` to
`None` after success. It does not require a square matrix. Subsequent inference
passes the full matrix to every component. This is **not** batch-style kernel
subsetting or a general streaming-kernel implementation. After multiclass
pairwise batch `fit`, components that validate width normally reject
`partial_fit` because they were trained on fewer columns. Two-class fits retain
all columns and can transition successfully. A failed update preserves the
previous indices here, consistent with this library's transactional updates.

Updates use the existing parallel helper and independent copies of components;
component copies must own their mutable learned state. A failed call leaves
prior learned state intact, whereas sklearn can leave partially updated state.
At least two distinct declared classes are required: the pinned sklearn
instead reaches an `IndexError` for a single class. Invalid declarations raise
Mojo errors with descriptive messages, not Python exception objects. Like
`fit`, `partial_fit` mutates in place and returns a mutable borrow of the wrapper.

## Parallel fitting

`n_jobs` follows joblib's `Parallel` rules and affects `fit` and `partial_fit`:

| `n_jobs` | Workers |
| --- | --- |
| `None`, `1` | Components fit one after another on the calling thread |
| `k > 1` | Up to `k` MAX CPU worker threads, never more than there are components |
| `-1` | One per logical core; `-k` means all but `k - 1`, and at least one |
| `0` | `ValueError: n_jobs == 0 in Parallel has no meaning`, raised after `X` and `y` are validated and before any component fits |

Each task fits its own clone of `estimator` into a fixed slot, so
`estimators_` keeps its usual order and the fitted model is identical for
every `n_jobs` (for deterministic components). If several components fail, the
error of the first one in fit order is raised, as in a sequential fit, and the
previous model is kept. Prediction is sequential, as in sklearn.

## Parameters

`get_params(deep)` returns `OneVsOneParams[E.Params]` holding the component's
own parameter record plus `n_jobs`; `set_params` accepts the same record.
Keyword overloads `set_params(n_jobs=...)` and `set_params(estimator=...)`
cover partial updates. `clone` returns an unfitted wrapper around a clone of
the component.

## Differences from scikit-learn

- `n_jobs` runs components on threads that share memory, not in loky worker
  processes. `E.fit` and `E.partial_fit` must therefore be safe to run
  concurrently on independent components that read the same `X`; a component that mutates process-global state
  should be fitted with `n_jobs=None`. Negative values count
  `std.sys.num_logical_cores()`, which ignores the CPU-affinity and cgroup
  limits joblib honours, and `joblib.parallel_config` has no equivalent.
- The column-vector `y` `DataConversionWarning` is not emitted.
- Not yet supported: sparse inputs, `fit_params`/metadata
  routing and `feature_names_in_`. Unsupported targets raise; other cases fail
  to compile.

## Validation

`pixi run test` runs `tests/multiclass/test_ovo.mojo`, which covers:
- shapes, ordering and pair subsets
- the vote/confidence formula
- binary and probability-only components
- Float32/Float64 regressors, exact 0/1 votes, fractional predictions, method precedence and parallel refits
- ties and NaN features
- invalid inputs with exact sklearn messages, and unfitted errors
- failed and repeated fits, clone/params and weighted score

`tests/multiclass/test_parallel.mojo` adds parallel fits that must be bitwise identical to
sequential ones, and checks that the first failing pair's error is reported.

It also runs the compile-time rejections, for example `predict_proba` and a
mismatched feature dtype.

`pixi run parity` also runs `tests/multiclass/ovo_parity.py`. There, scikit-learn's own
`OneVsOneClassifier` wraps the Python twins of `tests/multiclass/ovr_fixtures.mojo`, over
65 cases:
- 2, 3, 4 and 5 classes
- decision-only, probability-only and dual classifiers, plus Float32/Float64 regressors
- integer and float labels
- unbalanced pair subsets
- exact ties and NaN inputs

Classes, component counts, `pairwise_indices_`, predictions and unweighted
scores must match exactly, and weighted scores within 1e-15 relative.
Decision values from margin components must agree within 1e-12 relative,
which covers FMA contraction in the fixture (observed ≤ 2e-15 absolute).
Probability-component decisions must agree within 5e-11 absolute, because the
fixture's Mojo `exp` differs from libm by up to ~1.2e-11 relative and the
transform scales that error down (observed ≤ 6e-14). These components are test
doubles, so parity covers the one-vs-one logic, not any particular base
classifier or regressor.

`tests/multiclass/test_ovo_partial_fit.mojo` adds incremental capability, pair lifecycle,
validation, failure-state, clone and parameter contracts. The compile-failure
probe checks that nonincremental components cannot call `partial_fit`.
`tests/multiclass/ovo_partial_fit_parity.py` compares 324 update checkpoints against
sklearn using the existing incremental signed-sum fixtures: 2/3/5 classes,
int64/float32/float64 labels, sequential and parallel updates, missing pairs,
column targets and transitions in both directions between `fit` and
`partial_fit`. It also compares rejected calls from fresh, fitted and
incremental states. Counters, labels and predictions match exactly; decision
values allow 1e-14 relative / 1e-15 absolute for confidence-division rounding.
These are wrapper contracts using test doubles, not an incremental learner port.

`tests/multiclass/test_pairwise.mojo` checks the capability, exact Cartesian subsets and
binary targets, Float32/64 and strided gathers, parallel ordering, shape errors,
failed fits, configuration changes, refits, clones, strings and incremental
transitions. `tests/multiclass/pairwise_parity.py` generates 84 seeded fixtures against
the pinned sklearn source, including probability-only components, numeric and
string labels, two to five classes, and OvR/OutputCode forwarding. It compares
all gathered training values, targets, indices and predictions exactly.
Confidence divisions and weighted scores allow 1e-14 relative / 1e-15 absolute
rounding. The recording components are test doubles; this validates wrapper
routing rather than implementing or establishing parity for SVC.

## String labels

Select `kind=StringLabels` and pass shared `StringTarget` values. For example:

```mojo
from moxie.multiclass import OneVsOneClassifier, StringLabels, StringTarget

var model = OneVsOneClassifier[MyBinaryClassifier, kind=StringLabels](MyBinaryClassifier())
var y = StringTarget(["north", "south", "north"])
model.fit(X, y)
var predicted = model.predict(X)  # StringTarget, rank one
```

`classes_` is an owned `List[String]` in sorted Unicode order. Predictions are
rank-one `StringTarget` values; an `(n, 1)` training column can be supplied as
`StringTarget(values, n, 1)`. Empty strings and non-ASCII labels are supported.
Multi-column string targets are rejected. Component targets remain numeric
binary codes. Decision/probability columns follow `classes_` order where those
methods are available. Cloning, refitting, parallel fitting and weighted accuracy
have the same semantics as with numeric labels.

The existing `label=DType...` parameter continues to select numeric label dtype
by default. `kind` selects the target representation; the implementation's
`ClassValues` and `Domain` parameters must retain their defaults.

With incremental components, `partial_fit` also accepts string targets. Pass
`classes=StringTarget([...])` on the first call. Later declarations must contain
the same sorted unique classes, and batch labels must belong to that domain.

The component target ABI remains `NDArray[int64]` during string `partial_fit`.
Pinned sklearn's OvO helper uses `zeros_like(y)`, giving components empty-string
/ `"1"` targets when `y` is a string array, while still declaring integer binary
classes. Moxie passes integer 0/1 consistently. The string incremental oracle
normalizes those reference component targets explicitly; it tests the wrapper's
class mapping and updates, not identical component input dtypes.
