# Coordinate residual engine

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
The optional `random_seed` is a per-solve UInt32 seed for shared
`our_rand_r`. When provided, each sweep samples active positions with
replacement for as many draws as there are active features.
At alpha=0 the pinned reference's ridge dual formulation or OLS gradient norm
is used. Initial convergence may return zero iterations. At an iteration limit
the result remains available with converged=False. After a final failed gap
check the residual solver applies Gap Safe screening before returning the
coefficients, as in the pinned implementation. Invalid inputs or nonfinite
final coefficients/gap raise. Inputs, including initial coefficients, are never
mutated. Strided inputs are materialized once as contiguous feature columns.

`coordinate_descent_gram` is the cyclic solver for precomputed systems. Supply
a contiguous symmetric Gram matrix, a matching cross-product vector, owned
initial coefficients, and the squared target norm. The arrays use the same
Float32 or Float64 dtype. The routine supports the same unnormalized penalties
and optional positivity, and can check the gap before the first sweep with
`early_stopping=True`. Sparse coding supplies one Gram matrix for every input
row and solves each target separately.

See [scope](scope.md) for unfinished branches and validation commands. The
engine does not center inputs, normalize weights, derive solve seeds, fit an
intercept or perform CV; the estimator/family owners supply those operations.
