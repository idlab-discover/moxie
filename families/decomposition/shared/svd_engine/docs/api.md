# Dense SVD engine

```mojo
from moxie.decomposition import randomized_svd

var factors = randomized_svd(x, 2, random_state=42)
# factors.U: (rows, 2), singular_values: (2), Vt: (2, columns)
```

Run `pixi run mojo run -I builds/deps examples/randomized.mojo` after building.
The example constructs a small dense matrix. Arrays use
`numojo.NDArray[DType.float32]` or `NDArray[DType.float64]` without promotion.
Inputs are borrowed and unchanged. Results own their storage, and copying a
result makes independent array copies. Valid positive/negative-stride views
are accepted. NuMojo `.copy()` may copy a larger backing allocation, while
transpose allocates. Concurrent mutation of inputs or RNG is unsupported.

## randomized_svd

The integer-seeded overload accepts `x, n_components` followed by required
keyword `random_state: Int`. The stream overload accepts a third positional
`mut RandomState`. Integers reset at each call; a mutable stream advances.
Use one application-owned entropy context for reference `None` semantics.

| Keyword | Default | Meaning |
| --- | --- | --- |
| n_oversamples | 10 | Extra Gaussian columns; nonnegative |
| n_iter | -1 | Native sentinel for reference `"auto"`; otherwise nonnegative |
| power_iteration_normalizer | `"auto"` | `none` for <=2 iterations, LU otherwise; explicit `none`, `QR`, `LU` supported |
| transpose | `"auto"` | Wide matrices transpose automatically; typed alternatives `"true"`, `"false"` |
| flip_sign | True | Make first maximal-absolute U loading positive, flipping paired Vt |
| svd_lapack_driver | `"gesdd"` | `"gesdd"`: native one-sided Jacobi; `"gesvd"`: native Golub-Kahan bidiagonal QR |

Auto iterations are 7 for ranks below 10% of the smaller matrix dimension,
otherwise 4. Gaussian draws follow C traversal and consume all requested
oversampling columns, even when workspace rank exceeds matrix dimensions.
Requested ranks exceeding the smaller dimension return the available economy
rank, as in the reference helper. Inputs must be finite, nonempty rank-two
arrays. Errors use descriptive Mojo `Error` prefixes.

The native implementation uses shared RNG and rank-two NuMojo matrix products,
Householder QR, rectangular partial-pivot LU normalization, and one-sided
Jacobi SVD (`gesdd`) or Golub-Kahan-Reinsch bidiagonal QR SVD (`gesvd`) of the
projected matrix. It does not link LAPACK: the driver names select the native
algorithm closest to each LAPACK driver's method, so results agree with SciPy
to rounding, not bit-for-bit.
Float32 intermediates in matmul/QR/LU retain their dtype; scalar factorization
arithmetic and Jacobi workspace widen to Float64, then return same-dtype arrays.
Subspace or reconstruction comparisons are required at repeated/zero singular
values; their individual vectors are not uniquely defined.

## Private reusable functions

Import these from `moxie.decomposition._svd_engine`:

- `dense_svd(x, flip_sign=True, max_sweeps=100)` returns economy factors with
  all min(rows, columns) singular values descending and nonnegative. Per-column
  scaled norms and orthogonal null-space completion handle
  conditioning; representability limits still apply.
- `golub_kahan_svd(x, flip_sign=True, max_iterations=75)` returns the same
  economy factors by Householder bidiagonalization and implicit-shift QR
  (Golub-Reinsch 1970), the method family of LAPACK gesvd. It scales the input
  by its largest magnitude only, so it keeps less dynamic range than the
  per-column-scaled Jacobi kernel; exhausting the QR budget raises LinAlgError.
- `truncated_svd(x, n_components)` selects leading factors deterministically.
  It computes an economy SVD first; it is not an iterative or faster solver.
- `symmetric_eigh(x, max_sweeps=100)` requires exact symmetric square input,
  returns descending signed eigenvalues and column eigenvectors. It never clips
  negative eigenvalues; PSD policy belongs to the estimator.
- `qr_basis(x)` returns economic orthogonal Q, completing deficient directions.
- `randomized_range_finder(x, mut rng, size=..., n_iter=...,
  power_iteration_normalizer="auto")` returns a sampled orthogonal basis.
- `svd_flip(mut U, mut Vt, u_based_decision=True)` implements first-index tie
  decisions. False selects each Vt row instead. Factors must have disjoint
  writable storage; this in-place helper does not diagnose shared-buffer aliasing.
- `center_matrix(x)` returns an owned `(centered, means)` tuple using the shared
  accepted NuMojo mean and broadcast subtraction. It is separate from SVD;
  randomized_svd itself never centers.

Exact factors certify orthogonality and reconstruction; randomized factors
certify the appropriate projected equation (different for transpose=True),
not equality of their truncated approximation with the input. Certificates cost
O(m*n*k + (m+n)*k*k), and currently run on every call. This correctness baseline
makes no competitive performance claim. Symmetric Jacobi costs O(n^3) per sweep;
economy SVD costs O(m*n^2) per sweep for tall input and transposes wide input.
