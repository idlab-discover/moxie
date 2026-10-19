# Mixture models

Moxie owns `GaussianMixture` and `BayesianGaussianMixture` in
`src/moxie/mixture/`. Public imports execute the concrete estimators. Their
shared EM driver, initialization, responsibilities, precision factors and
Gaussian statistics/log densities have one implementation here.

```mojo
from moxie.mixture import GaussianMixture, BayesianGaussianMixture

var gaussian = GaussianMixture[DType.float64](2, random_state=0)
_ = gaussian.fit(X)
var labels = gaussian.predict(X)
var bayesian = BayesianGaussianMixture[DType.float64](n_components=2, random_state=0)
_ = bayesian.fit(X)
```

Both support dense Float32/Float64 NuMojo arrays, all four covariance layouts,
all four initialization methods, multiple restarts, warm starts, zero-iteration
fits, prediction, responsibilities, likelihood scores and sampling. Gaussian
also supports explicit initial arrays and AIC/BIC. Bayesian supports both weight
prior variants and data-dependent Gaussian-Wishart priors; it has no AIC/BIC.

Read the [Gaussian API](gaussian/api.md), [Bayesian API](bayesian/api.md),
[Gaussian scope](gaussian/scope.md), [Bayesian scope](bayesian/scope.md), and
[shared contracts](plumbing.md). `GaussianMixtureParams` and
`BayesianGaussianMixtureParams` are public typed configuration records. Result
types remain available as estimator associated types, including `Samples`.
Caller-owned streams use `moxie._random.RandomState` with `fit_with_rng`,
`fit_predict_with_rng` and `sample_with_rng`.

## Validation

From Moxie, run `pixi run mixture-build`, `pixi run mixture-test`, and
`pixi run mixture-parity`. Root `build`, `test`, and `parity` include these
sources and checks. Compiler flags are `-O3 --fp-mode=contract=off`; the lock
retains Mojo 1.0.0 and MAX 26.5.0. Python, NumPy, SciPy and sklearn are validation
and tooling dependencies only.

Tests cover the original family contracts and both child suites, concrete public
imports with generic fluent dispatch, independent fitted copies, two isolated
consumer assemblies, numerical ownership, missing-source/export conflicts,
Gaussian-only information criteria and executable examples. Examples are
`examples/gaussian_mixture.mojo` and `examples/bayesian_mixture.mojo`.

Parity executes the local sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` and verifies the support wheel's
KMeans sources against that checkout. Reports, including exception cases and
reference lower bounds, are written separately to
`builds/mixture/gaussian/parity/report.json` and
`builds/mixture/bayesian/parity/report.json`. The existing comparison tolerances,
reduction order, random traversal and special functions are preserved.
Gaussian's Float32 ill-conditioned dataset remains excluded because its second
moments cannot represent the small variances. Divergent and drifting Float32
trajectories are reported separately from strict comparison passes.
`tests/mixture/known_float32_exceptions.json` restricts exceptions to the cases
observed before relocation; new exceptional cases fail the parity command.

Benchmark tooling remains at `tests/mixture/{gaussian,bayesian}/benchmark.py`;
run with `pixi run python` after building. It writes under `benchmarks/mixture/`.
No performance claim follows from the migration checks.

## Numerical ownership and consumer migration

`DEPENDENCIES.json` reads the tested numerical sources live from
`../.worktrees/mojo-linalg-mixture`, branch `feat/mixture-digamma`, whose migration
baseline is `314f0b1`. This is provenance, not a source pin. The assembler maps
the needed factorization closure and `special_gamma` into
`moxie._mixture_linalg`. Source files remain owned by the numerical checkout;
Moxie does not vendor them. The ownership check verifies their assembled bytes.

Moxie's existing `mojo_linalg` namespace still comes from the covariance
worktree. Its Cholesky and indefinite-factor logarithms differ from mixture's
version. The private namespace prevents collisions or silent substitution.
Shared `RandomState` sampling continues using the existing `mojo_linalg` SVD.
At migration, its `svd`, `_spectral` and `_storage` files are byte-identical to
the mixture checkout, preserving that tested sampling implementation.
Mixture imports Cholesky, triangular solve, symmetric eigendecomposition and
`NumericalError`; Bayesian additionally uses digamma, gammaln and betaln.
Numax does not replace these kernels during this migration. Local k-means++ and
Lloyd initialization remain in `_kmeans_init`; there is no cluster dependency.

External manifests should consume Moxie's current root manifest and source
paths. Remove `mixture`, `mixture-gaussian` and `mixture-bayesian` projects that
point to the old sibling and use Moxie's public exports. Preserve the explicit
`mixture-linalg` contribution. `_api` and `_api_types` were scaffold declarations
and are not part of the implemented package. No active Moxie path needs the
retired sibling. Shared framework, dense storage and RNG continue to come from
their existing owners.

Python reflection, metadata routing, pandas output, feature names, pickling,
array-API dispatch, sparse inputs and other exclusions in the estimator guides
remain unsupported. Sample points are Float64 for both estimator dtypes.
Full/tied sample coordinates can differ from NumPy because the SVD orientation
can differ; parity checks labels, draw consumption and Mahalanobis norms.
