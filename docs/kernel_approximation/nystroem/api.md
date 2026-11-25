# Dense Nystroem

Import `Nystroem`, `NystroemKernelParams` and `NoCallableKernel` from
`moxie.kernel_approximation`. See [the example](../../../tests/kernel_approximation/nystroem/example.mojo).

| Constructor | Keywords and defaults |
| --- | --- |
| `Nystroem[dtype=DType.float64, Kernel=NoCallableKernel, output_dtype=dtype](kernel="rbf", ...)` | `gamma: Optional[Float64]=None`, `coef0=None`, `degree=None`, `kernel_params: Optional[NystroemKernelParams]=None`, `n_components: Int=100`, `random_state: Optional[Int]=None`, `n_jobs: Optional[Int]=None` |
| `Nystroem[dtype, K](K(...), ...)` | callable kernel: `n_components`, `random_state`, `n_jobs` |

Named kernels are sklearn's `PAIRWISE_KERNEL_FUNCTIONS` (`additive_chi2`,
`chi2`, `linear`, `poly`, `polynomial`, `rbf`, `laplacian`, `sigmoid`,
`cosine`) plus `precomputed`, with sklearn's defaults (gamma 1/n_features, chi2
gamma 1.0, degree 3, coef0 1). `fit` draws `RandomState.permutation(n)` and keeps
its first `k = min(n_components, n_samples)` entries as `component_indices_`
(Int64), copies those rows into `components_` `(k, p)`, evaluates the basis
Gram with the kernel_ridge engine, factors it with mojo-linalg and stores
`normalization_ = U / sqrt(max(S, 1e-12)) @ Vt` `(k, k)`. `transform(X)`
returns `K(X, components_) @ normalization_.T` `(n, k)`. The output dtype matches
the input except for sequential Float32 Laplacian, which uses Float64 (see below).

Language adaptations (tested in `tests/kernel_approximation/nystroem/contracts.mojo`):

- `kernel_params` is a typed `NystroemKernelParams(gamma, coef0, degree)`;
  explicit constructor values override it, unused entries are ignored.
- Callable kernels are typed structs implementing kernel_ridge's `RowKernel`
  (`evaluate(X, i, Y, j)`) and hold their own parameters; they are passed to
  the constructor overload and stored in `params.kernel_function` (kernel
  string `"callable"`). `kernel_params` with a callable raises; gamma/coef0/
  degree with a callable or precomputed kernel raise sklearn's error.
- `precomputed` requires `k == n_samples` (sklearn's shape error otherwise).
- sklearn's `n_components > n_samples` warning is recorded in `fit_warnings`
  (a native `List[String]`, not an sklearn attribute).
- With `random_state=None`, fitting draws the basis indices from the process-wide
  stream. Use `fit_with_rng` or `fit_transform_with_rng` with a mutable
  `RandomState` for a caller-owned stream.
- `n_jobs=None` or `1` evaluates sequentially; positive values cap MAX CPU
  workers, and negative values use `max(1, logical_cores + 1 + n_jobs)`.
  Workers are capped by row count. Zero raises. Independent row blocks use
  the shared kernel engine, then join before errors or learned state are
  published. Precomputed matrices need no kernel scheduling. Joblib contexts
  are not consulted. Callable kernels must be thread-safe when parallelized;
  the parallel path evaluates all pairs without the sequential symmetry shortcut.
- Mojo requires a static result type. Sequential Float32 Laplacian therefore uses
  `Nystroem[DType.float32, output_dtype=DType.float64]("laplacian", ...)`.
  Components retain Float32; kernel evaluation, normalization and transform
  use Float64. Omitting this output parameter for Float32 Laplacian raises.
  sklearn's parallel pairwise assembly casts Laplacian back to Float32, so
  effective `n_jobs != 1` requires the default `output_dtype=dtype`, even for
  a one-row input. All other kernels also require that default. Changing the
  runtime kernel or `n_jobs` with `set_params` checks this constraint before use.
  Negative job counts use logical cores, without joblib's cgroup/affinity policy.
- The kernel engine rejects nonfinite kernel values; sklearn would propagate them.
- Typed `get_params()`/`set_params`, `clone` (unfitted, copies the callable),
  `nystroem<k>` feature names.

Exactly symmetric RBF basis matrices use mojo-linalg's `symmetric_eigh` when
every eigenvalue clears the clipping floor by a roundoff margin. For these
positive-definite matrices the two singular bases coincide, so the symmetric
eigenbasis gives the same inverse square root. Other kernels, RBF matrices
that are asymmetric or have small or nonpositive eigenvalues, and failed
eigensolves use the economy one-sided Jacobi SVD. Both solvers work in Float64
and return factors in the input dtype. Numax requires compile-time matrix
dimensions for Array and CPU Tensor `eigh` and `svd`. Nystroem selects its basis
size at runtime.
For a rank-deficient basis, choices of singular vectors in the null space
can change entries of `normalization_` while preserving the feature Gram
`Z_B Z_B^T`.

## Validation

`pixi run parity` executes the unedited pinned sklearn methods and pairwise
kernel functions (revision 4358542a, support wheel for Cython helpers).
It covers named kernels, default/explicit parameters, tall/wide/tiny/duplicated
inputs, precomputed shapes, parameter/domain errors, seeds, parallel job counts,
and the notebook's 270-component digits RBF basis in both dtypes.
Indices and subsequent RNG draws must match exactly. Well-conditioned cases
bound normalization and transform error by `64 eps kappa max(1, max|ref|)`.
Laplacian promotion is checked at Float64 precision.

Ill-conditioned tests check feature Grams where numerically determined, plus
held-out transform composition with an explicit dot-product/kernel roundoff bound.
Severely ill-conditioned Float32 cases do not promise entrywise agreement with
LAPACK: clipping tiny singular values can amplify rounding substantially.
An independent analytic suite checks 36 diagonal spectra in both dtypes, with
scales 1e-8 through 1e8 and singular values above, at and below the 1e-12 floor,
including zero. Held-out rows excite every mode; feature Grams must match
`Y diag(s / max(s, 1e-12)) Y.T` within `128 eps max(1, max|expected|)`.
