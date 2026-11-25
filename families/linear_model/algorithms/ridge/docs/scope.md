# Ridge dense scope

`Ridge`, `RidgeClassifier` and `ridge_regression` support dense Float32 and
Float64 with auto, SVD, Cholesky, LSQR, CG, SAG, SAGA and positive L-BFGS-B.
The API supports scalar and per-target alpha, sample weights and vector or
matrix targets. Classification supports binary, multiclass and multilabel
inputs with balanced or explicit class weights.
`ridge_regression(return_intercept=True)` uses SAG to fit an intercept.

The [API](api.md) records result shapes, seed semantics and solver-specific
weight handling. Sparse inputs remain deferred. Native diagnostics omit
Python verbosity and input validation always runs.

Run `python families/linear_model/algorithms/ridge/tools/check.py test`, then
`python families/linear_model/algorithms/ridge/tools/check.py parity-only`
with the locked toolchain. Contracts cover both dtypes, solver dispatch,
repeat fits, clone, invalid inputs and the optional result fields. The parity
suite uses pinned `_ridge.py`, `_base.py` and the compiled SAG sources.
The script records tolerances and checks solver selection and epoch counts.
