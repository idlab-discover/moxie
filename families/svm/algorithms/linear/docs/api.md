# Linear estimators in Mojo

Import `LinearSVC`, `LinearSVR`, `ClassWeight`, `Dual`, and `l1_min_c` from
`moxie.svm`. Run `pixi run start` here for the numeric, string-label and regression
example. All estimator constructor names/defaults match the pinned reference,
with the native adaptations below. LinearSVR defaults to epsilon=0.0 and
loss="epsilon_insensitive"; LinearSVC defaults to penalty="l2" and
loss="squared_hinge"; both default to dual="auto", C=1, tol=1e-4, max_iter=1000,
fit_intercept=True, intercept_scaling=1, verbose=0.

```mojo
var model = LinearSVC(penalty="l1", dual=False, random_state=42)
model.fit(X, labels)
var prediction = model.predict(X)
var decision = model.decision_function(X)
model.set_params(C=2.0)
model.fit(X, labels)
```

Inputs use `numojo.NDArray[float64]` or `[float32]`, samples in rows. Outputs,
coefficients and intercepts are Float64. LinearSVC's default label type is
Float64: target, `classes_` and prediction are `List[Float64]`, each a vector.
`LinearSVC[String]` uses native string lists; `LinearSVC[Int]` uses integer lists.
Numeric NDArray targets also work with the default classifier and Float64 X.
LinearSVR targets/predictions are rank-one arrays; its `coef_` is `(p,)`, versus
LinearSVC's `(1,p)` or `(k,p)`. Binary decisions are `(n,)`; multiclass decisions
are `(n,k)`. Intercepts are always rank-one arrays, including the zero intercept
when fit_intercept=False (sklearn may use a scalar zero in that case).

`dual` accepts a Bool or "auto" through the typed `Dual` record. Class weights
use "none" (the native None default), "balanced", or
`ClassWeight[Label](labels, values)`. Missing mapping labels have weight one;
extraneous labels are ignored. Public configuration records own their arrays.

`sample_weight` is an optional Float64 vector. `LinearSVC` also implements
the `moxie.base.WeightedSupervisedEstimator` keyword fit contract for Float64
features and native list labels. Passing an existing array into
an Optional may need `.copy()` under Mojo's ownership rules:
`model.fit(X, labels, sample_weight=weights.copy())`. This explicit copy does
not turn the training data into a mutable alias. Models own learned arrays.
`get_params()` returns a typed record, `set_params(record)` or keyword updates
change configuration, and `moxie.base.clone(model)` returns an unfitted model.
Every fit overload, fit_with_random_state, and parameter setter returns a mutable
borrow of the original estimator. Chaining works with weighted and Float32
inputs too: `model.set_params(C=2.0).fit(X, labels).predict(X)`.
Use `_ = model.fit(X, labels)` when discarding the return value.
A validation failure retains the previous valid learned model.

`random_state=-1` represents the reference None default. A nonnegative integer
is a reproducible seed. `fit_with_random_state(X, y, rng, sample_weight)` borrows
and advances a shared `moxie._random.RandomState`; it consumes the same
randint(INT_MAX) seed draw as the reference before calling LIBLINEAR. An integer
configuration restarts its stream each fit. The None context is local to the
estimator. `verbose>0` prints an iteration summary. Both estimators expose
`fit_warnings: List[FitWarning]` under the
[native fit-warning contract](../../../../../docs/interfaces.md#fit-warnings).
Reaching `max_iter` records the reference `ConvergenceWarning` category and
message on the estimator. A converged refit clears the list, and a failed fit
preserves the previous record. `fit_status_` also exposes capped fits.

`l1_min_c(X, labels, loss="squared_hinge", fit_intercept=True,
intercept_scaling=1.0)` returns a Float64 bound. It accepts the same native label
lists. Use loss="log" for the L1 logistic bound; no logistic estimator is
exported by this child. See [scope.md](scope.md) for restrictions and evidence.

## Sample-weight requests

`LinearSVC` and `LinearSVR` use the shared
[sample-weight request API](../../../../../docs/interfaces.md#sample-weight-requests).
Configure fit and score independently through fluent setters:

```mojo
_ = classifier.set_fit_request(sample_weight="training_weight").set_score_request(
    sample_weight=True
)
var requests = classifier.get_metadata_routing()
```

`True` requests `sample_weight`, `False` declines it, and `None` resets the
request to unset. A string requests a weight by an ASCII identifier. Omitting
the keyword retains the current request. The getter returns an owned copy, and
configuration clones retain both requests. Native setters are always available.
Parents can use the returned requests to resolve metadata. Direct calls to
`fit` and `score` use the explicitly supplied weights.

## Feature names

Pass `feature_names=` with a `List[String]` to associate names with the
columns of a dense array. The list must have one entry per column. Names
can repeat and their order is significant.

```mojo
var names: List[String] = ["height", "width"]
var predictions = classifier.fit(X, y, feature_names=names).predict(
    X, feature_names=names
)
```

Both estimators accept the keyword on `fit`, `predict`, and `score`.
`LinearSVC.decision_function` accepts it too. Named inference checks the
column count and, when the fit recorded names, their exact order. Arrays
carry no column labels. Mixing named and unnamed calls is allowed without
the Python warnings. The fitted estimator owns its
`feature_names_in_: Optional[List[String]]`. A successful unnamed fit
clears it, a failed fit preserves it, and an unfitted clone starts with
`None`.

## Native model files

Use the experimental [model file API](../../../../../docs/serialization.md)
with a schema from `moxie.svm.linear_serialization`.

```mojo
from moxie.serialization import load_model, save_model
from moxie.svm.linear_serialization import LinearSVCSchema, LinearSVRSchema

save_model[LinearSVCSchema[String]](classifier, "classifier.moxie")
var restored = load_model[LinearSVCSchema[String]]("classifier.moxie")

save_model[LinearSVRSchema](regressor, "regressor.moxie")
var restored_regressor = load_model[LinearSVRSchema]("regressor.moxie")
```

Choose the same classifier label type on save and load. Schema version 3
supports String, Int, Int32, Int64, Float32, and Float64 labels, including
explicit class weights. Both schemas preserve current configuration, fitted or
unfitted state, and the estimator-owned random stream. Fitted loads restore
independent coefficients and intercepts without retraining. Settings changed
after fit apply to a subsequent fit. A stream borrowed through
`fit_with_random_state` is owned by the caller.
Version 3 preserves fitted feature names and both sample-weight requests.
Versions 1 and 2 load with names unset. Version 1 also leaves both requests
unset. All three versions reconstruct `fit_warnings` from the saved fit
status.

The linear child test runner includes file round trips and malformed-state
checks. The linear wrapper parity driver serializes and restores each
successfully fitted estimator before comparing with pinned scikit-learn.

The linear wrapper parity test also compares named weighted fits against
scikit-learn DataFrames for Float32 and Float64. It checks exact name order
and classifier predictions, then compares margins, regression predictions,
and weighted scores at the wrapper tolerances. Native contract tests check
name ownership, failed fits, unnamed refits, reordered names, and archive
validation.
