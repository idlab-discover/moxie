# Ridge dense scope

Implemented and validated: `Ridge`, `RidgeClassifier`, `ridge_regression` on
dense Float32/Float64 with solvers auto/svd/cholesky (primal, dual, fallbacks),
lsqr (with `n_iter_`), sparse_cg, lbfgs (positive=True); scalar or per-target
alpha, nonnegative sample weights, vector/matrix targets, class_weight
(balanced / explicit), binary/multiclass/multilabel classification.

Checks: `tests/contracts.mojo` (solver agreement, positivity, shapes,
per-target alpha errors, weights vs repetition, kernel path, invalid
solver/positive/lbfgs/sag/alpha/max_iter, unfitted errors, clone, classifier
classes/multiclass/multilabel/class weights). `tests/parity.py`: 149 cases vs
pinned sklearn `_ridge.py`/`_base.py`, tolerances documented in the script
(observed max 2.2e-14 Float64, 2.4e-6 Float32), exact `n_iter_` and `solver_`.

Unfinished: `sag`/`saga` (need the SAG engine; tracked in the family handoff).
Deferred (dense-only scope): sparse X including the sparse intercept paths.
Not native: `verbose`, `random_state` (only used by sag), `check_input=False`.
