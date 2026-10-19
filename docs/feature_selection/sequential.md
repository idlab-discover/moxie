# Sequential feature selection

`SequentialFeatureSelector` greedily adds or removes features using mean
cross-validation scores. It requires fitting and scoring capabilities, not
coefficients or feature importances. Each fold fits a fresh clone through the
shared `model_selection.cross_val_score` engine. The supplied estimator is
preserved; the selector does not refit a final estimator or expose prediction.

```mojo
from moxie.linear_model import Lasso
from moxie.feature_selection import (
    SequentialFeatureSelector, DenseSupervisedCV, SequentialCount, SequentialCV,
)

comptime CV = DenseSupervisedCV[Lasso[DType.float64]]
var selector = SequentialFeatureSelector[CV](
    Lasso(alpha=0.08),
    n_features_to_select=SequentialCount(2),
    direction="forward",
    cv=SequentialCV(5),
)
selector.fit(X, y)
var selected = selector.transform(X)
var original_columns = selector.get_support(True)
```

Float32 models use `DenseSupervisedCV[Lasso[DType.float32], DType.float32]`.
ElasticNet works through the same adapter. These two real estimators are covered
by numerical parity tests. Other dense supervised estimators can use the adapter
when they implement the foundation SupervisedScorer and Copyable contracts with
matching X and sample-weight array types, and an independently typed copyable
target (StringTarget or Int64/Float32/Float64 array). No family importance trait
is required. Estimator-specific restrictions, including the adopted Lasso and
ElasticNet solver options, still apply.

Constructor keywords retain the reference defaults:

| Parameter | Behavior |
| --- | --- |
| `n_features_to_select="auto"` | A `SequentialCount` containing `"auto"`, a positive integer below the feature count, or a Float64 fraction in `(0, 1]` |
| `tol=None` | In auto mode, stop when the new mean score improves by less than tol. Negative values are allowed for backward selection; forward selection accepts zero or positive values. Ignored for stopping with an explicit count/fraction, but still validated |
| `direction="forward"` | Add features; `"backward"` removes them |
| `scoring=None` | Use the adapter's default scorer; alternatively pass an instance of its typed Scorer |
| `cv=5` | Native `SequentialCV`: integer/None or an owned snapshot of shared `IndexSplit` pairs |
| `n_jobs=None` | Nonzero integers accepted for executed searches; the adopted CV engine currently executes serially |

The native constructor specializes on a `SequentialAdapter` type. Its optional
second positional argument supplies an adapter instance; otherwise its default
constructor is used. `SequentialCount`, `SequentialCV`, typed scorer records and
parameter records replace Python unions, callable reflection and dictionaries.
Methods mutate in place and do not return self.

Without tol, auto performs `p // 2` greedy steps. This retains `p // 2` columns
in forward mode and `p - p // 2` columns in backward mode, including the pinned
reference's difference for odd feature counts. With tol, auto performs at most
`p - 1` steps. Fractions are truncated: a small positive fraction can select zero
features in forward mode; backward mode may consequently attempt a zero-column
fit and fail, as in the reference. A fraction of 1.0 selects all columns.
Zero-step requests do not execute folds, so they can succeed even when the
dataset has fewer rows than the configured fold count.

Candidates are visited in original column order. Equal scores retain the first
candidate. NaN comparisons preserve Python max behavior, including a first
candidate whose mean is NaN. Shared CV fills failed-fold scores with NaN and
raises if every fit fails. Python warning transport is unavailable.

Default CV uses unshuffled KFold for regression/unsupervised targets and
StratifiedKFold for classifiers with binary/multiclass targets. Classifier
recognition comes from the adapter; DenseSupervisedCV uses ClassifierMixin and
shared label encoding. To use another splitter, materialize its shared
IndexSplit pairs and pass `SequentialCV(pairs)`. The configuration deep-copies
those indices and reuses them for every candidate. Stateful splitter objects
that generate new folds on each candidate call are not exposed.

`fit_named(X, y, names, fit_metadata=..., score_metadata=...)` supports feature
names and explicit metadata routing through the shared CVAdapter. The supplied
dense adapter treats metadata as optional sample-weight arrays and slices them
by training/test indices. Pass `Optional[NDArray[dtype]]` values for these two
arguments. The engine permits distinct fit and scoring metadata. Python
`**params`, routing configuration, scorer-name lookup and container output
settings are not exposed.

Custom adapters extend `SequentialAdapter`, which extends the shared CVAdapter.
They supply dtype, default_scorer, empty_metadata and the inherited fit/row
slicing methods. Optional hooks are cv_labels, is_classifier, allows_nan and
missing_target. The input must be an NDArray of the declared dtype; the target
and metadata types are otherwise adapter-owned. An unsupervised adapter can
supply missing_target so `fit(X)` and `fit_transform(X)` work. This extension is
tested with explicit control-flow fixtures; the shipped adapters cover supervised regressors and DecisionTreeClassifier.
The dense adapter takes its target type from the estimator independently of
feature dtype, preserving StringTarget and Int64 labels in rows and stratification.
See [classifier selection](classifier-selection.md). Custom scorers must be Copyable;
estimators must implement BaseEstimator and Copyable. Pairwise-kernel column
semantics need a dedicated adapter and are not supplied here.

`support_()`, `n_features_to_select_()`, `n_features_in_()` and
`feature_names_in_()` are learned-attribute accessors. Support, transforms,
inverse transforms and feature names use the family SelectionState. Outputs
are owned; strided inputs are supported. Failed refits preserve the last valid
selection. `get_params()` returns SequentialParams; `set_params(params)` changes
configuration without discarding fitted support. Cloning copies configuration
and clones the estimator into an unfitted selector. It never fits the original
estimator. Sparse input remains deferred.

Run `pixi run build`, `pixi run test` and `pixi run parity`. The sequential parity
harness loads the original class from scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` with installed sklearn 1.9.1 validation,
CV and estimator infrastructure. Set `MOXIE_REFERENCE_PYTHON` to such an
interpreter if the existing workspace oracle environment is unavailable.

The 64 real-model cases cover both dtypes and directions, count/fraction/auto
selection, tolerance stopping, negative strides, multi-output regression,
explicit folds and routed weights. Support/count comparisons are exact; fold
scores use rtol=5e-5/3e-8 for Float32/Float64 and atol=rtol/10, accounting for
coordinate-descent and reduction rounding. Native contracts separately exercise
cloning, names, ownership, parameter changes, failure behavior, classifier
stratification, unsupervised adapters, ties, NaNs and invalid inputs. Both
isolated consumer assemblies run real sequential selection. This evidence does
not establish parity for every possible estimator, scorer or Python API.
