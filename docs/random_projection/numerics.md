# Numerical implementation decisions

This integration retains `ProjectionMatrix` and the existing projection and
pseudoinverse helpers as private random-projection implementations. It does not
introduce a shared sparse backend.

The trait path takes NuMojo `NDArray` input. `project_dense` multiplies a strided
NDArray by the `ProjectionMatrix` components with the same input-dtype
accumulation order as `project`, finite input and result checks, and one result
allocation. Numax's dynamic `numax.linalg.blas.matmul` takes TensorLike buffers
and dispatches to MAX GEMM. It would require materializing dense components
from the row-major list or CSR buffers and would change the reduction order.
`numojo.NDArray.__matmul__` likewise needs dense NDArray components. For CSR
components, both approaches would allocate a dense matrix and compute products
for its implicit zeros.

`project` retains input-dtype accumulation and the dense/CSR output rule.
Both paths use the stored component order and validate public component buffers.

The existing scaled one-sided Jacobi pseudoinverse works in Float64 and applies
`max(shape) * input_dtype_epsilon * largest_singular_value`. It handles tall,
wide, singular and zero components, detects nonconvergence and validates results
before returning. It densifies components and uses O(k*d + min(k,d)^2) workspace.

Alternatives inspected during integration:

- Shared `moxie._linalg.covariance_inverse` requires square positive-definite
  symmetric input and rejects singular matrices, so it cannot replace this call.
- Numax `linalg.basic.pinv` requires compile-time known tall dimensions. Its
  Array-tier `pinv` uses a different default rank threshold and fixed-sweep SVD
  behavior. Neither is a direct replacement for the runtime shapes and tested
  rank/convergence contract here.
- `mojo_linalg.svd.svd` supports runtime rectangular NuMojo inputs, but returns
  factors in the selected dtype. Replacing the widened workspace with Float32
  factors would alter rounding. Widening into an NDArray and building the inverse
  from those factors requires conversion buffers and separate cutoff/scale tests.
  The existing implementation is retained for this migration; this is not a claim
  that the shared backend cannot support a future replacement.
- `std.math.log` on the locked compiler loses enough precision at 100 to change
  a small-epsilon integer JL bound. The existing `log64` and its exact bound
  fixture remain. No new logarithm implementation was added.

Sampling now exclusively uses `moxie._random.RandomState`, including its
`normal`, `binomial` and `sample_without_replacement` methods. The old copied
MT19937 implementation is removed. Both parity suites cover integer-seeded
coefficients; mixed-stream fixtures check Gaussian cache retention across sparse
sampling and exact draw consumption. Global and explicit streams have native
contracts. No shared numerical or RNG implementation changes are required.
