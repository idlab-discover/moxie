# Estimator reference contracts

These are the reference contracts for the implemented dense estimators.
See the algorithm API guides for native signatures and explicit adaptations.
The oracle is sklearn commit `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Constructor defaults

The following signatures are extracted from the pinned `_classes.py`; preserve
all options in typed native parameter records before implementing each child.
`self` below is the Python reference receiver, not a required Mojo argument.

```text
LinearSVC(self, penalty='l2', loss='squared_hinge', *, dual='auto', tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
```

```text
LinearSVR(self, *, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual='auto', verbose=0, random_state=None, max_iter=1000)
```

```text
SVC(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability='deprecated', tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```

```text
NuSVC(self, *, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability='deprecated', tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```

```text
SVR(self, *, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
```

```text
NuSVR(self, *, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
```

```text
OneClassSVM(self, *, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
```

`l1_min_c(X, y, *, loss="squared_hinge", fit_intercept=True,
intercept_scaling=1.0)` also accepts loss="log". SVC/NuSVC's deprecated probability
sentinel is effectively false; typed records must represent sentinel/false/true
without silently changing this default. Deprecated calibration fields are retained. Native errors carry category prefixes;
Python warning and deprecation machinery are not reproduced.

## Methods and learned state

| Estimator | Dense methods | Learned state / shapes |
| --- | --- | --- |
| SVC, NuSVC | fit(X,y,sample_weight), predict, decision_function, score; predict_proba and predict_log_proba only with probability=True | sorted classes_ `(k,)`, class_weight_ `(k,)`; support_ `(s,)`, support_vectors_ `(s,p)`, n_support_ `(k,)`, dual_coef_ `(k-1,s)`, intercept_ `(q,)`, n_iter_ `(q,)`, fit_status_, shape_fit_, n_features_in_; deprecated probA_/probB_ `(q,)` when enabled |
| SVR, NuSVR | fit(X,y,sample_weight), predict, score | support arrays as above, n_support_ `(1,)`, dual_coef_ `(1,s)`, intercept_ `(1,)`, scalar n_iter_, fit_status_, shape_fit_, n_features_in_ |
| OneClassSVM | fit(X,y=None,sample_weight), fit_predict, predict, decision_function, score_samples | support arrays and scalar iterations as regression, offset_ `(1,)`, intercept_=-offset_; labels ±1 |
| LinearSVC | fit(X,y,sample_weight), predict, decision_function, score | sorted classes_, coef_ `(1,p)` binary or `(k,p)` multiclass, intercept_ `(1,)` or `(k,)`, scalar n_iter_ (max across classes), n_features_in_ |
| LinearSVR | fit(X,y,sample_weight), predict, score | coef_ `(p,)`, intercept_ `(1,)`, scalar n_iter_, n_features_in_ |

Kernel `coef_` is available only for the linear kernel; it is derived from support
vectors/dual coefficients and is read-only. Callable/precomputed models follow
reference support-vector storage and index semantics. Training precomputed input
is `(n,n)`, inference `(m,n)`; no symmetry/positive-definiteness test is imposed by
the implementation. Kernel predictions are `(m,)`; classifier probabilities `(m,k)`;
binary decision `(m,)`, multiclass OVO `(m,q)` or OVR `(m,k)`, q=k(k-1)/2.
Single-class training is rejected for classifiers, but is valid for OneClassSVM.
Regression and one-class have no public decision_function inherited from BaseSVC
except OneClassSVM's explicitly supplied method; avoid accidental extra traits.

Classifier score is weighted accuracy; regression score is weighted R², including
constant-target behavior. OneClassSVM has no classifier/regressor score contract.
No estimator supports incremental fitting or warm starts. All expose typed
parameter access and unfitted configuration cloning; copying learned arrays is
a separate operation. Native fit/set_params return a mutable borrow.
Feature-name metadata, reflection, metadata routing, pandas, Python serialization,
Python callable objects and estimator repr are future binding work.

## Dense acceptance obligations for each child

Every dense kernel (linear/poly/rbf/sigmoid/precomputed/native callable), scale,
auto and numeric gamma, shrinking on/off, cache limits, tolerance/iteration
limits, class/sample weighting, sorted label encoding, OVO/OVR/tie behavior,
probability calibration and RNG consumption remain in scope where reference
supported. Nu wrappers retain feasibility checks and reuse C-SVM presentation.
LIBLINEAR covers the solver IDs, loss/penalty/dual combinations and
Crammer-Singer contract in plumbing.md; intercept regularization and dual=auto
require dedicated comparisons. LinearSVC dense L1 coefficients remain in scope.

Children test Float32-to-Float64 conversion, contiguous/strided inputs, finite
validation, nonempty dimensions, target length/rank, nonnumeric class labels,
weights (including zero/negative masked samples), zero-variance gamma, degenerate
kernels, large values, infeasible nu, one-class targets, convergence warnings,
unfitted access, repeat fit, parameter updates, clone independence and deterministic
seeds. Use the pinned source/tests, not assumptions from older sklearn releases.
Tests must check public shapes/dtypes, sign conventions and class/pair ordering.
The native callable contract and owned cache return require explicit tests and
cost accounting in their engine sessions. Document tolerance per solver/problem;
compilation and exact small-value checks alone do not establish parity.
