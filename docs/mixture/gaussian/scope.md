# mixture.gaussian scope (dense only)

Implemented: `GaussianMixture` for Float32 and Float64 dense NuMojo arrays, all
four covariance types, all four initializations (KMeans via a local
k-means++/Lloyd translation), `weights_init`/`means_init`/`precisions_init`,
`n_init` restarts, `warm_start`, `max_iter=0`, convergence reporting,
predict/predict_proba/score_samples/score/sample/bic/aic, typed params/clone.
Shared kernels for the Bayesian mixture: EM driver (`_em`), stable
responsibilities, precision factors, Gaussian sufficient statistics and log
densities, initialization.

Sparse input: none in `sklearn.mixture` (dense `validate_data`); nothing is
deferred and no sparse input is densified.

Validation (`pixi run mixture-test`, `pixi run mixture-parity`):
- `test`: kernel regressions (logsumexp stability and -inf rows, precision
  factor round trips, SPD/positivity errors, statistics, all initializers) and
  estimator contracts (shapes per covariance type, both dtypes, invalid
  parameters, unfitted errors, input errors, init-array validation, collapsed
  data error, repeat-fit determinism, clone independence, set_params, borrowed
  RNG advancement, max_iter=0, nonconvergence warning, warm start, strided
  input, ownership of learned arrays) and the example.
- `parity`: executes the pinned sklearn `mixture/_base.py` and
  `_gaussian_mixture.py` (4358542a) against the native runner on blobs, tall,
  wide, tiny and ill-conditioned data, every covariance type x init, two seeds
  and n_init=2, lifecycle branches (nonconvergence, max_iter=0, warm start,
  reg_covar, init arrays) and collapsed-data errors, in Float64 and Float32.
  Pre-rollup recorded result: 221 strict numerical comparisons and 16 error-path checks pass, with 3 additional Float32 divergences reported separately. Tolerances:
  Float64 rtol 1e-9 / atol 1e-11 (1e-6 / 1e-6 for the tiny and 1e5-scale-ratio
  ill datasets, where near-singular covariances amplify summation-order
  rounding through log-determinants); Float32 rtol 5e-4 / atol 1e-3 (2e-2 /
  2e-3 tiny); BIC/AIC 50x rtol (they scale the mean log-likelihood by n);
  labels, n_iter_, converged_ exact. Float32 cases seeded from single rows or
  near-symmetric random responsibilities may follow a different (often
  permuted, equally good) trajectory: 3 cases are reported as diverged, with
  finite native fits and both lower bounds in `builds/mixture/gaussian/parity/report.json`.
  The Float32 ill-conditioned dataset is excluded (its second moments are not
  representable in Float32).

Performance baseline: `pixi run python tests/mixture/gaussian/benchmark.py` (single thread) writes
`benchmarks/mixture/gaussian/baseline.json`; these timings are historical and are not a migration performance claim.
