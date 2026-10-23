# Gaussian processes

Moxie owns dense `GaussianProcessRegressor` and the implemented
`moxie.gaussian_process.kernels` API. `GaussianProcessClassifier` remains
unimplemented. The retired Gaussian-process sibling is not a dependency.

- [Regression API](regression/api.md) and [scope](regression/scope.md).
- [Kernel API](kernels/api.md) and [scope](kernels/scope.md).
- [API coverage and deferred features](api-surface.md).
- [Native contracts and ownership](plumbing.md).

Sources live in `src/moxie/gaussian_process/`, validation in
`tests/gaussian_process/`, and runnable examples in `examples/gaussian_process.mojo`
and `examples/gaussian_process_kernels.mojo`.

Side-by-side notebooks against the official gallery live in
`notebooks/gaussian_process/`. `plot_gpr_prior_posterior` covers prior and
posterior moments and `sample_y` for five kernels. `plot_gpr_noisy` covers two
local optima, an 80 x 80 likelihood grid, seeded restarts, `make_pipeline`,
`cross_val_score`, `GridSearchCV` and `clone`. Rerun them with
`pixi run -e notebooks notebooks notebooks/gaussian_process`. Ill-posed fits
(RationalQuadratic with `alpha` bounds up to 1e15, DotProduct**2 with a rank-3
covariance) end at different L-BFGS-B points in Moxie and sklearn because
float64 rounding drives the optimizer. The notebooks show that the posteriors
agree at a common theta and that Moxie is at least as accurate there.

From Moxie, run `pixi run build`, `pixi run test`, and `pixi run parity`.
For focused validation, run `pixi run python tools/gaussian_process_checks.py test`
and `pixi run python tools/gaussian_process_checks.py parity`.
The same suites run through the root commands. The reference remains
scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`; the existing Mojo
1.0.0 / MAX 26.5.0 lock is unchanged.

## Numerical ownership

The root manifest consumes live sources from these external owners:

| Source | Assembled namespace | Purpose |
| --- | --- | --- |
| `../.worktrees/mojo-linalg-gaussian_process` | `moxie._gp_linalg` | Original GP factorizations and general-nu Matérn Bessel functions |
| `../mojo-optimize` | `moxie._optimize` | Original bounded L-BFGS-B behavior |
| `shared/random` | `moxie._random` | Shared default state, integer seeds, borrowed streams and sampling |

The GP linalg worktree remains required. Its `special.mojo` is absent from the
main linalg checkout, and its Cholesky module differs from the covariance
worktree. A separate assembled namespace preserves its source bytes without
replacing Moxie's existing covariance/linear-model dependencies. Optimizer
sources also remain separate from the linear-model optimizer worktree.
The manifest and consumer checks enforce these owners; no numerical library
implementation is copied into the Gaussian-process package.

## Compatibility

Float32 inputs are widened for Float64 computation and output. Both values of
`copy_X_train` retain independent training copies. Parameter updates preserve
learned state until refitting and invalid updates restore configuration.
Likelihood evaluation with `clone_kernel=False` can change learned kernel theta
without replacing the cached posterior factorization.

Integer random seeds reproduce seeded behavior; omitted estimator state and
explicit `None` sampling advance the shared stream. Caller-owned streams advance
in place. Omitted `sample_y` state is integer zero. Sampling parity compares
whitened norms because valid SVD sign choices can change individual draws.

Classification, structured/custom kernels, callable pairwise metrics, training
aliasing, Python reflection, metadata routing and pickling remain deferred.
CompoundKernel theta splitting preserves the documented correction to the
reference setter; unsupported nested CompoundKernel compositions remain explicit.
