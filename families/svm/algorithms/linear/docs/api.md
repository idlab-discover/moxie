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

`sample_weight` is an optional Float64 vector. Passing an existing array into
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
estimator. `verbose>0` prints an iteration summary, not LIBLINEAR's per-iteration
C logging. A ConvergenceWarning message and `fit_status_` expose capped fits.

`l1_min_c(X, labels, loss="squared_hinge", fit_intercept=True,
intercept_scaling=1.0)` returns a Float64 bound. It accepts the same native label
lists. Use loss="log" for the L1 logistic bound; no logistic estimator is
exported by this child. See [scope.md](scope.md) for restrictions and evidence.
