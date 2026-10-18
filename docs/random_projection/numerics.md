# Numerical implementation decisions

This integration retains `ProjectionMatrix` and the existing projection and
pseudoinverse helpers as private random-projection implementations. It does not
introduce a shared sparse backend or an NDArray adapter.

`moxie._dense` accepts NuMojo NDArray storage and supplies finite rank-2
multiplication, but no CSR operations. `numojo.NDArray.__matmul__` would require
copying the existing public list buffers and cannot preserve the CSR path that
sums duplicate entries without densifying inputs. `project` retains input-dtype
accumulation, finite-result validation and the exact dense/CSR output rule.
It allocates result storage; it does not allocate a dense copy of CSR inputs.

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
