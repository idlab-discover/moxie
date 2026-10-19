# OneVsRestClassifier

`moxie.multiclass.OneVsRestClassifier` ports `sklearn.multiclass.OneVsRestClassifier`
for dense single-label targets and multilabel indicator matrices. It fits cloned
binary classifiers or regressors: one per label, or one for a single-label binary
problem. Multiclass prediction selects the highest score; multilabel prediction
thresholds each label independently. Parity is checked against the pinned scikit-learn revision in
[family.md](family.md).

```mojo
from moxie.multiclass import OneVsRestClassifier

var model = OneVsRestClassifier(MyBinaryClassifier(), n_jobs=None, verbose=0)
model.fit(X, y)                       # X: NDArray[float64] (n, p), y: NDArray[int64] (n,)
var labels = model.predict(X)         # NDArray[int64], values from classes_
var scores = model.decision_function(X)  # only if the component has decision_function
var proba = model.predict_proba(X)       # only if the component has predict_proba
```

See [examples/multiclass_one_vs_rest.mojo](../../examples/multiclass_one_vs_rest.mojo) for a runnable program.

## Type parameters and the component contract

`OneVsRestClassifier[E, dtype=DType.float64, label=DType.int64]`:

| Parameter | Meaning |
| --- | --- |
| `E` | `SupervisedEstimator & Predictor & Copyable`, conforming to `RegressorMixin`, or to `ClassifierMixin` with `DecisionFunction` and/or `ProbabilityPredictor` |
| `dtype` | Feature dtype; `E.Input` must be `NDArray[dtype]` |
| `label` | Class label dtype: `int64`, `float32` or `float64` |

`E.Target` must be `NDArray[DType.int64]`: each component receives one 0/1
column, exactly as sklearn's `LabelBinarizer` produces. Component decision
values must be `NDArray[float32/float64]` of shape `(n,)` or `(n, 1)`, and
probabilities `(n, 2)`. Regressor predictions must be `NDArray[float32/float64]`
of shape `(n,)` (an `(n, 1)` column is also flattened). Incompatible dtypes
fail to compile; malformed score shapes raise `ValueError`.

## Behavior

| Item | Behavior (matches sklearn unless noted) |
| --- | --- |
| `classes_` | Sorted unique labels, or column indices `0..n_labels-1` for multilabel; `NDArray[label]` |
| `estimators_` | `List[BinaryComponent[E]]`: one per label; single-label binary fits use one component |
| `n_classes_()` | `len(classes_)` (1 after a one-class fit) |
| `multilabel_()` | `True` after fitting an indicator matrix, otherwise `False` |
| `n_features_in_` | `Optional[Int]`, recorded from rank-two `X` |
| `predict` | Argmax of component scores, first class wins ties; NaN scores propagate like `np.maximum`. Two classes: strict threshold 0.5 for regressors; for classifiers, 0.0 on `decision_function`, else 0.5 on P(positive) |
| `decision_function` | `(n,)` for one component, else `(n, K)`; widened to Float64 |
| `predict_proba` | `(n, K)` positive-class probabilities, rows normalized when nonzero only for single-label targets; `(n, 2)` for one component |
| `score` | Mean accuracy, or exact-match subset accuracy for multilabel; optionally weighted |
| One-class fit | sklearn's `_ConstantPredictor`: `BinaryComponent.is_constant()`, predicts the single class, probability `[1, 0]` |

For regressor components, a prediction score comes from `predict(X)`, even
when the component also provides `decision_function`. For classifier components,
it comes from `decision_function` when available, otherwise from
`predict_proba(X)[:, 1]`. The wrapper remains a classifier: its `score` computes
accuracy, and each component is trained on integer 0/1 targets. A regressor must
therefore accept `NDArray[int64]` targets. Public `decision_function` and
`predict_proba` retain their component-method semantics and compile-time
availability; neither method is synthesized from regressor predictions.

### Dense multilabel targets

Pass an `NDArray[label]` of shape `(n_samples, n_labels)` containing 0/1,
with at least two label columns. Each column is fitted independently; a sample
can receive zero, one or several labels. `predict` returns a dense 0/1 array
of the same label dtype and column count. `decision_function` and
`predict_proba` return `(n_samples, n_labels)` arrays. Probabilities are marginal
per-label probabilities and are **not row-normalized**. `score` counts a sample
as correct only when every label matches; weights apply to whole samples.

```mojo
# Y: NDArray[int64], e.g. rows [1, 0, 1], [0, 1, 1], ...
model.fit(X, Y)
var indicators = model.predict(X)  # (n_samples, n_labels)
var subset_accuracy = model.score(X, Y)
```

Always-absent and always-present columns use constant components with decision
values and probabilities 0 and 1 respectively. As in sklearn, the prediction
threshold is selected from the **first fitted component** for all columns; a
constant first component selects 0.5, including when later components use
margins. Otherwise the usual classifier/regressor threshold applies.

An `(n, 1)` target retains sklearn's single-label behavior. Empty targets and
non-0/1 indicator values are rejected. Unlike sklearn's permissive target
inference, nonstandard two-valued matrices (e.g. 0/2) are not binarized.
`classes_` and predictions retain the compile-time `label` dtype, whereas
sklearn uses integer column indices and integer indicator predictions even
for floating-point indicator inputs. Sparse targets remain unsupported.

Fit validates the target like `LabelBinarizer` (via
`moxie._labels`): continuous, NaN, multioutput and out-of-int64-range float
labels raise `ValueError`. An `(n, 1)` column target is accepted and flattened.
State is committed only after every component fits; a failed fit leaves the
previous model intact (sklearn may leave partial state).

## Single-label incremental fitting

`partial_fit(X, y, classes=None)` is available when `E` implements
`IncrementalClassifier` with `Classes = NDArray[DType.int64]`. The wrapper also
conforms to `IncrementalClassifier`; its `Classes` type is `NDArray[label]`.

```mojo
var model = OneVsRestClassifier(MyOnlineBinaryClassifier())
model.partial_fit(X_first, y_first, classes=all_classes.copy())
model.partial_fit(X_next, y_next)
```

On the first call, supply a nonempty rank-one array containing every possible
class. Classes are sorted and deduplicated. A batch may omit any class; later
`classes` arguments may reorder or repeat labels but must describe the same
set. Unknown batch labels are rejected. Targets may be rank one or `(n, 1)`.

Each component receives binary targets and `classes=[0, 1]` on **every** update.
There is one component for one or two declared classes, otherwise one per class.
Unlike ordinary `fit`, incremental batches never substitute constant predictors:
even an all-zero or all-one column updates a real component. A one-class problem
sends all-zero targets. Components retain learned state across calls; replacing
the prototype through `set_params` affects future clones, not existing components.
Normal `fit` replaces incremental state, and `clone` returns an unfitted wrapper.
Updates after a single-label `fit` are supported unless that fit created a
constant component, which has no incremental method.

Multilabel updates require an ordinary indicator `fit` first, with both 0 and
1 present in every column so that every component supports incremental updates.
Subsequent indicator batches must retain all columns in their original order;
a label absent from a batch is an all-zero column, not a removed column.
All-zero and all-one batches update the existing components normally. Optional
class declarations describe the same column indices `0..n_labels-1` and may
repeat or reorder them. A constant component created by `fit` rejects any
update that selects it.

Fresh indicator `partial_fit` calls are rejected, even with classes supplied:
the pinned sklearn implementation fits its label binarizer on declared classes,
which establishes single-label mode. Updates cannot change a single-label model
to multilabel mode. Dense target binarization reuses Moxie's preprocessing
`label_binarize`; it does not require a local preprocessing implementation.

After a multilabel `fit`, label-vector batches are also accepted when their
values belong to `classes_`. They are binarized against that class set while
`multilabel_` stays true. For exactly two labels, the reference produces one
binary column and drops the second component; `predict` then raises because
the component count no longer matches the class count. Decision values remain
rank one, and probabilities have two columns. Later indicator batches update
only the retained component. A new ordinary `fit` restores the full component
set. Prefer consistently shaped indicator batches when training a multilabel problem.

Incremental regressors and metadata routing remain unsupported.

Updates run on copies of learned components and install results only after all
updates succeed. As with ordinary fit, this differs from sklearn's possible
partial state after an error. Components must provide independent fitted copies
for this guarantee; copies with explicitly shared mutable storage cannot be
rolled back. Component validation governs feature-count consistency.

## Parallel fitting

`n_jobs` follows joblib's `Parallel` rules and affects `fit` and `partial_fit`:

| `n_jobs` | Workers |
| --- | --- |
| `None`, `1` | Components fit one after another on the calling thread |
| `k > 1` | Up to `k` MAX CPU worker threads, never more than there are components |
| `-1` | One per logical core; `-k` means all but `k - 1`, and at least one |
| `0` | `ValueError: n_jobs == 0 in Parallel has no meaning`, raised after the target is validated and before `X` is checked or any component fits |

Each task fits its own clone (or updates a fitted copy) into a fixed slot, so
`estimators_` keeps its usual order and the fitted model is identical for
every `n_jobs` (for deterministic components). If several components fail, the
error of the first one in fit order is raised, as in a sequential fit, and the
previous model is kept. Prediction is sequential, as in sklearn.

## Parameters

`get_params(deep)` returns `OneVsRestParams[E.Params]` holding the component's
own parameter record plus `n_jobs` and `verbose`; `set_params` accepts the same
record. Keyword overloads `set_params(n_jobs=..., verbose=...)` and
`set_params(estimator=...)` cover partial updates. `clone` returns an unfitted
wrapper around a clone of the component. As in sklearn, a negative `verbose`
is accepted on assignment and rejected with `ValueError` when `fit` or
`partial_fit` starts.

## Differences from scikit-learn

- `n_jobs` runs components on threads that share memory, not in loky worker
  processes. `E.fit` and `E.partial_fit` must therefore be safe to run concurrently
  on independent clones or fitted copies that read the same `X`; a component
  that mutates process-global state should be fitted with `n_jobs=None`. Negative values count
  `std.sys.num_logical_cores()`, which ignores the CPU-affinity and cgroup
  limits joblib honours, and `joblib.parallel_config` has no equivalent.
- `verbose` is validated and cloned but prints nothing.
- `decision_function`/`predict_proba` availability is decided at compile time
  from `E`. When the first component is constant, sklearn may expose methods
  absent on the prototype (and fail on later nonconstant components); here
  those methods stay unavailable.
- Not yet supported: sparse
  inputs, `fit_params`/metadata routing,
  `label_binarizer_` and `feature_names_in_`. Unsupported targets raise; other
  cases fail to compile. The "Label not x is present in all training examples"
  warning is not emitted.

## Validation

`pixi run test` runs `tests/multiclass/test_ovr.mojo` (shapes, ordering, binary and
one-class cases, ties, invalid inputs, unfitted errors, refits, failed fits,
clone/params, weighted score, multilabel subset accuracy and transitions
between single-label and multilabel fits) and compile-time capability rejections.
`tests/multiclass/test_parallel.mojo` checks joblib's `n_jobs` resolution, that parallel
fits are bitwise identical to sequential ones across repeated runs, where the
`n_jobs=0` error falls in the validation order, and that the first failing
component's error is reported.
`pixi run parity` runs `tests/multiclass/ovr_parity.py`: scikit-learn's own
`OneVsRestClassifier` wraps Python twins of the test components in
`tests/multiclass/ovr_fixtures.mojo`, and 181 cases (1, 2, 3 and 5 classes; decision-only,
probability-only and dual classifiers; Float32/Float64 regressors, including
regressors with a distinct decision method; integer and float labels; exact ties;
NaN inputs and binary threshold boundaries; 2, 3 and 5 multilabel columns,
constant columns, Float32/Float64 indicators and multilabel threshold boundaries) must match. Classes, component
counts, predictions and unweighted scores must match exactly. Decision values must agree within 1e-12 relative,
which covers FMA contraction. Probabilities must agree within 5e-11 relative,
because the fixture's Mojo `std.math.exp` differs from libm by up to ~1e-11.
These components are test doubles: parity covers the one-vs-rest logic, not any
particular base classifier.

`tests/multiclass/test_ovr_partial_fit.mojo` covers class declarations, missing batch classes,
state retention, clone/reset behavior, invalid updates, and multilabel transitions.
`tests/multiclass/ovr_partial_fit_parity.py` compares 324 batch checkpoints against sklearn:
one, two, three and five classes; all supported label dtypes; sequential and
parallel updates; column targets; and updates following ordinary fit. It compares
component update/sample/positive counts as well as classes, output shapes,
predictions, decisions and probabilities. Quarter-grid arithmetic is exact;
probabilities allow 1e-14 relative / 1e-15 absolute rounding error. The components
are paired test doubles, not production incremental classifier implementations.

Incremental parity also checks multilabel fit-to-update transitions, all-zero
and all-one batches, vector transitions, two-label component truncation, and
reference rejections for fresh indicators, changed widths/classes and constant
components. The two-label vector transition also checks the reference prediction
error after component truncation.

## String labels

Select `kind=StringLabels` and pass shared `StringTarget` values. For example:

```mojo
from moxie.multiclass import OneVsRestClassifier, StringLabels, StringTarget

var model = OneVsRestClassifier[MyBinaryClassifier, kind=StringLabels](MyBinaryClassifier())
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

## Dense precomputed inputs

OvR forwards the full input matrix to each binary component during fitting
and inference; it needs no per-pair indices. It implements `PairwiseInput`
and delegates `pairwise_input()` to the template estimator, matching sklearn's
pairwise tag. Components without that capability default to non-pairwise.
See [the OvO capability contract](one_vs_one.md#dense-precomputed-kernels).
Square-kernel and prediction-width validation are the component's responsibility.
The pairwise parity suite checks full-matrix forwarding and tag propagation
against pinned sklearn using recording components.
