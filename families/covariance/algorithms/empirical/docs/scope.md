# Dense scope and status

Implemented (dense Float32/Float64): EmpiricalCovariance, ShrunkCovariance,
empirical_covariance, shrunk_covariance (single and batched), log_likelihood,
and the inherited methods get_precision, mahalanobis, score and error_norm
(frobenius and spectral, scaling/squared options) for both precision cache
modes and both centering modes.

Validation: `pixi run covariance-test` runs native contracts (lifecycle, unfitted errors,
clone/set_params, failed-refit preservation, feature mismatch, NaN input,
invalid shapes/parameters, singleton and rank-deficient covariances, stable
log determinants, spectral norm, strided views, batched shrinkage, engine
trait dispatch). `pixi run covariance-parity` compares 2726 quantities over 54 seeded
datasets (tall, square, wide, 1x1, 2x1, single sample, constant column,
ill-conditioned, large offset; Float64 and Float32) with the pinned sklearn
sources; tolerances are derived in `tests/parity.py`.

Sparse branches in these sklearn APIs: none. CSR/CSC/COO input is deferred by
the workspace SPARSITY-SCOPE decision; nothing densifies sparse input.

Known cost: the pseudoinverse uses mojo-linalg's cyclic Jacobi eigensolver;
for p=200 a fit with store_precision=True takes about 3.3 s versus 28 ms
without the precision (see ignored benchmarks/baseline.json). A faster
symmetric eigensolver belongs in mojo-linalg.

Run these commands from the Moxie repository root.
