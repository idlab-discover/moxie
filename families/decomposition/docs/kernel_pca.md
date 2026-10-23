# Dense KernelPCA

`KernelPCA[DType.float32|float64]` embeds finite dense samples in a centered
kernel eigenspace. Float64 is the default. The public constructor follows the
local sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

```mojo
from moxie.decomposition import KernelPCA
var model = KernelPCA(3, kernel="rbf", gamma=0.2, fit_inverse_transform=True)
var embedding = model.fit_transform(X)
var test_embedding = model.transform(X_test)
var preimage = model.inverse_transform(test_embedding)
```

Run [the example](../examples/kernel_pca.mojo) and native contracts with
`MOXIE_ESTIMATORS=kernel_pca pixi run test`; differential checks use
`MOXIE_ESTIMATORS=kernel_pca pixi run parity`.

Supported kernels are linear (default), poly, rbf, sigmoid, cosine and
precomputed. Gamma defaults to 1 / training feature count, also for the inverse
map. Precomputed fit accepts a square training Gram matrix and transform accepts
(test samples, training samples). Inverse fitting rejects precomputed kernels.
The inverse is a kernel ridge pre-image approximation, not an exact inverse.
It fits the uncentered kernel on training embeddings with diagonal `alpha`;
non-positive-definite systems raise, with no minimum-norm fallback.

`dense` computes exact leading eigenpairs. `arpack` and `auto` use the same
exact dense computation: this is a documented algorithmic adaptation, as in the
family's PCA/SVD; tol and max_iter are validated but do not control Lanczos
iterations. `randomized` uses the shared randomized SVD, with signed singular
values following sklearn's `_randomized_eigsh`. Integer seeds are repeatable;
None uses shared entropy. `iterated_power` accepts an integer or `"auto"`.
No sparse input containers are supported.

Eigenvalues are descending, significantly negative values raise, and the
reference's dtype-specific PSD thresholds clip small values. Eigenvector signs
use the first maximal absolute entry. `remove_zero_eig=True` removes zero
columns; `n_components=None` always removes them. An all-zero kernel can produce
(samples, 0) outputs. Repeated eigenvalues define a subspace, not a unique basis.

Learned arrays `eigenvalues_`, `eigenvectors_`, `X_fit_`, and (when inverse fitting
is enabled) `X_transformed_fit_` and `dual_coef_` are Optional owned NuMojo arrays.
`gamma_` and `n_features_in_` record training metadata. Fit replaces state only
on success; clone retains constructor configuration and has no learned buffers.
Strided inputs are accepted and copied. Inputs and prior fitted snapshots are
not mutated. Typed parameter records and named set_params updates are supported.

Mojo restrictions: `copy_X=False` raises because fitted snapshots must be owned;
callable kernels, Python metadata routing and non-native output containers are
unavailable. Named kernels ignore kernel_params as sklearn does. n_jobs is
accepted (zero is invalid); kernels execute through their native owner's backend.
`fit` mutates in place and returns a borrow of the estimator, so
`model.fit(X).transform(X)` chains as in sklearn (docs/interfaces.md, fluent fit).

Kernel operations are consumed from the current Moxie KernelRidge numerical owner;
no kernel implementation is copied here. Runtime eigen and Cholesky operations
use the existing mojo-linalg dependency, since Numax's static factorization API
and Mojo 1.1 toolchain do not fit this Mojo 1.0 runtime-shape contract.

The 55 differential cases cover four kernels, all solver choices, default gamma,
rank clipping, zero and repeated spectra, precomputed input, held-out transforms and inverse
maps. Float64 tolerance is 2e-7 and Float32 8e-4 for bounded fixtures; inverse
comparisons allow four times those tolerances for propagation through the solve.
Positive nondegenerate eigenvector signs/order are compared directly. These
checks establish the tested dense modes, not full sklearn compatibility or a
performance claim.
