# mixture.bayesian scope (dense only)

Implemented: `BayesianGaussianMixture` for Float32/Float64 dense NuMojo arrays
with Dirichlet-process and Dirichlet-distribution weight priors, Gaussian-Wishart
priors for full/tied/diag/spherical covariances, data-dependent prior defaults,
prior validation, variational updates, expected log weights/densities, the
lower bound, all initializations, restarts, warm start, max_iter=0, sampling.
The EM lifecycle and Gaussian kernels are consumed from the shared Moxie mixture modules.
Sparse input: none in `sklearn.mixture`; nothing deferred or densified.

Validation: `pixi run mixture-test` (priors, shapes, both dtypes, parameter/prior errors,
unfitted errors, monotone lower bound, determinism, clone/set_params, max_iter=0,
nonconvergence, warm start, borrowed RNG, agreement with GaussianMixture under a
data-dominated prior, example). `pixi run mixture-parity` executes the pinned sklearn
`_bayesian_mixture.py` (4358542a) on blobs/tall/tiny data for both prior types x
covariance types x initializations, custom priors, lifecycle branches and prior
errors. The pre-rollup record reports 234 strict numerical comparisons and 20 error-path checks passing, with 6 additional Float32 drifts reported separately. Float64: rtol 1e-9 / atol 1e-11
(tiny: 1e-6 / 1e-6); Float32: rtol 5e-4 / atol 1e-3, n_iter_ within 2. Six
Float32 cases drift along flat directions of the mixed-precision reference and
are recorded in `builds/mixture/bayesian/parity/report.json` with lower bounds within 1%; every
Float64 case matches strictly. Benchmark: `pixi run python tests/mixture/bayesian/benchmark.py` ->
`benchmarks/mixture/bayesian/baseline.json` (single thread).
