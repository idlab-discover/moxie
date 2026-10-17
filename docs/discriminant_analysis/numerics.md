# Numerical ownership

The family consumes Moxie's live `DEPENDENCIES.json` through
`shared/assembly/tools/assemble_family.py`. It does not recover historical
framework or numerical source snapshots. NuMojo owns dense array storage;
Mojo-linalg supplies SVD, least squares, symmetric eigendecomposition,
Cholesky and triangular solves. Those dependencies already belong to Moxie's
assembly. No new numerical runtime dependency is introduced.

Empirical covariance, fixed shrinkage, column means, and Ledoit-Wolf now
reuse `moxie.covariance` and its existing helpers on Float64 work arrays.
The family keeps row-major centering because the covariance helper returns
column-major work arrays. It retains StandardScaler's corrected variance and near-constant
feature test for automatic shrinkage, per-class aggregation, and its typed
covariance-estimator adapter. That adapter takes a private Float64 copy and
preserves the configured estimator. It is distinct from the public covariance
family's estimator trait.

The generalized symmetric eigenproblem composes the existing Cholesky,
triangular solve and symmetric-eigen kernels. The shared `_linalg` covariance
adapter requires exact symmetry in both triangles and reports `ValueError`;
this family reads the lower triangle and reports contextual `LinAlgError`.
Keeping this composition preserves its contract without another factorization
engine. Tall SVD retains its R-only unpivoted Householder reduction.
Mojo-linalg's `qr` forms Q and pivots columns; substituting it would add a
potentially large allocation and require undoing the permutation. The family
only needs singular values and right singular vectors.

Compact Float64 work arrays preserve the existing reduction order in remaining
family-specific products and class statistics. Valid strided Float32/Float64
inputs are materialized without retaining aliases; outputs use the estimator
dtype. Shared label encoders consume numeric NDArrays or `StringTarget`, whereas
this API accepts `List[L]` for arbitrary `ClassLabel` types. Its small generic
sorted-label adapter remains local.

See the [shared storage policy](../../shared/numerics/docs/numerics.md) and
[shared solve contract](../../shared/linalg/docs/linalg.md). The legacy
`../session-plan/NUMERICS.md` path is absent in this workspace; the current
Moxie owner documents define the consumed contracts.
