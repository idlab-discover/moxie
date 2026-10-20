# Cyclic residual engine

Import `coordinate_descent` from `moxie.linear_model._cd_fast` in the verified
assembly. This is a private engine, not a public sklearn path API.

`coordinate_descent(X, y, initial, alpha, beta, max_iter, tol, positive=False)`
accepts same-dtype Float32/Float64 NuMojo arrays: X (n,p), y (n,), initial (p,).
X/y must already be centered and square-root-weighted if needed. Alpha and beta
are **unnormalized** penalties: the engine minimizes half the squared residual
sum plus alpha times L1 norm plus beta/2 times squared L2 norm. Penalties and
tolerance must be finite and nonnegative; max_iter must be at least one.

`CoordinateResult` contains owned `coef`, `dual_gap`, `iterations` and
`converged`. The gap is unnormalized; the stopping threshold is tol times y@y.
At alpha=0 the pinned reference's ridge dual formulation or OLS gradient norm
is used. Initial convergence may return zero iterations. At an iteration limit
the result remains available with converged=False. Invalid inputs or nonfinite
final coefficients/gap raise. Inputs, including initial coefficients, are never
mutated. Strided inputs are materialized once as contiguous feature columns.

See [scope](scope.md) for unfinished branches and validation commands. The
engine does not center inputs, normalize weights, draw random numbers, fit an
intercept or perform CV; the estimator/family owners supply those operations.
