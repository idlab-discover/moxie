# LinearRegression dense scope

Implemented and validated: dense Float32/Float64 `LinearRegression` with
`fit_intercept`, `copy_X` (inputs never mutated), `tol`, `n_jobs` (accepted,
sequential), `positive` (NNLS), nonnegative `sample_weight`, vector and matrix
targets, `coef_`, `intercept_`, `rank_`, `singular_`, `n_features_in_`,
`predict`, weighted R2 `score`, params/clone/repeat-fit lifecycle.

Checks (`pixi run build|test|parity`):
- `tests/contracts.mojo`: unfitted/feature-count/NaN errors, minimum-norm tiny
  case (rank 1, singular sqrt(2)), exact intercepts, vector vs one-column vs
  multi-target shapes, zero/integer weights vs dropping/repeating samples,
  negative/mismatched weights, failed fits preserving state, exact rank
  deficiency, NNLS multi-target, strided views, clone independence, both dtypes.
- `tests/parity.py`: 38 cases (19 datasets x 2 dtypes) against the pinned
  sklearn `_base.py` source: tall, wide, multi-target, weighted (with zero
  weights), positive (plain/weighted), exact rank deficiency, cond~1e7 with and
  without a rank cut, tiny, single sample, column target, constant feature;
  with and without intercept. Tolerances are documented in the script
  (Float64 1e-8, Float32 3e-4; observed max 2.2e-14 / 2.4e-6).

Deferred by the dense-only scope (session-plan/SPARSITY-SCOPE.md): sparse
CSR/CSC/COO X and the sparse LSQR branch; no implicit densification.
Not native: Python-only `n_jobs` parallelism, metadata routing, feature names.
