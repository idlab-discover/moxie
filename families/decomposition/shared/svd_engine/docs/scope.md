# Dense decomposition scope

This project implements a native dense engine, not decomposition estimators.
Both `svd_lapack_driver` spellings are implemented natively: `"gesdd"` uses
one-sided Jacobi SVD (not LAPACK divide-and-conquer) and `"gesvd"` uses
Golub-Kahan-Reinsch bidiagonal QR iteration. Results match the reference to
rounding and within the validated tolerances; bit-identical LAPACK output is
not claimed.

## Supported paths

- Public `moxie.decomposition.randomized_svd` on finite nonempty rank-two
  NuMojo Float32/Float64 arrays, returning owned `SVDResult`.
- Seeded range finding with `auto`, `none`, `QR`, and pivoted `LU` power
  normalization; automatic/explicit transpose and optional paired sign flips.
- Private reusable economy `dense_svd`, deterministic `truncated_svd`,
  `symmetric_eigh`, `qr_basis`, `randomized_range_finder`, `svd_flip`, and
  `center_matrix` in `moxie.decomposition._svd_engine`.
- Rank-deficient, zero, singleton, tall, wide, repeated-value and strided input
  cases; independent owned outputs and borrowed mutable RNG streams.

The exact API and language adaptations are in [api.md](api.md). Float64 kernel
accumulation is deliberate; returned arrays retain input dtype. Residual and
orthogonality certificates reject invalid results. Large unnormalized powers
can overflow and raise; callers can select QR or LU normalization. Failure to
converge raises `LinAlgError`; no partial factors are returned.

## Deferred sparse and binding capabilities

CSR/CSC/COO storage, sparse range products, sparse iterative solver paths and
sparse input conversions are deferred by session-plan/SPARSITY-SCOPE.md.
Reference branches: `sklearn/utils/extmath.py` randomized_svd and
randomized_range_finder accept sparse inputs; decomposition `_pca.py` and
`_truncated_svd.py` invoke sparse iterative representations. This engine offers
only typed dense input and never densifies a sparse user object. Inherited
foundation and family contracts supply no sparse storage.

No sparse-dependent default must be overridden for dense randomized_svd.
Integer/complex
array promotion, Python array-API dispatch, implicit global random state and
Python object bindings are unavailable typed interfaces. For None-like RNG,
pass an application-owned mutable context as documented by the shared RNG owner.
No estimator, transform/fit lifecycle, or public randomized_eigsh is claimed.

## Checks

Executed commands: `pixi run build`, `pixi run test`, `pixi run parity`.
All three pass, including 917 numerical cases covering both SVD drivers.
Sparse import rejection also passes.
Build compiles the owned module, examples, both dtypes and two independently
assembled consumers. Test checks shape/rank/finite-input errors, views and
ownership, sign ties, seed reset versus mutable streams, driver agreement and
invalid driver names, and a compile-failing sparse import. Parity executes the pinned local sklearn
extmath function bodies and compares deterministic factors with SciPy.
Final executed results are recorded in the attempt handoff and ignored logs.
