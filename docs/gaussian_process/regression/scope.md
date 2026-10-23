# Dense GaussianProcessRegressor scope

Supported: dense Float32/Float64 X `(n, d)`, y `(n,)`/`(n, t)`, every kernel of the
kernels engine, scalar/per-sample alpha, normalize_y, L-BFGS-B with restarts,
optimizer=None, native custom optimizer, prior/posterior mean/std/cov, sample_y,
LML with gradients, R² score, clone/params. Sparse input is deferred
(`../session-plan/SPARSITY-SCOPE.md`); non-vector (structured) inputs are not
supported.

Validation: `pixi run test` (15 native contract tests + example), `pixi run
parity` (15 cases incl. Float32, multi-target, restarts, general-nu Matern,
no-optimizer, tiny data, against the pinned `_gpr.py`; 255 checks; tolerances:
optimized theta 1e-4 rel (Float32 2e-3, general-nu Matern 1e-3), derived
quantities 10x that, prior 1e-12, samples via whitened norm 1e-4),
The native benchmark driver is retained under `tests/gaussian_process/regression/`.
