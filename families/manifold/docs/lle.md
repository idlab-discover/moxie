# Dense locally linear embedding

`LocallyLinearEmbedding` and the array entry point `locally_linear_embedding`
implement standard LLE, modified LLE, Hessian LLE, and local tangent space
alignment. Fit learns `embedding_`, `reconstruction_error_`, `n_features_in_`,
and `nbrs_`. Transform projects each new row using its barycenter weights
against the fitted neighbors and the corresponding rows of the learned embedding.
It supports the same projection for all four methods.

```mojo
from moxie.manifold import LocallyLinearEmbedding, locally_linear_embedding

var model = LocallyLinearEmbedding(n_neighbors=12, method="modified", eigen_solver="dense")
var training_coordinates = model.fit_transform(X)
var query_coordinates = model.transform(query)
var result = locally_linear_embedding(X, n_neighbors=12, n_components=2)
print(result.reconstruction_error)
```

The estimator accepts Float32 and Float64 NDArrays, including strided views.
It owns a Float64 training snapshot and returns Float64 coordinates and
reconstruction error, like sklearn's estimator. The function accepts an NDArray
and returns a `LocallyLinearEmbeddingResult` with named fields `embedding` and
`reconstruction_error`. It also computes in Float64, including when sklearn's
functional standard LLE uses Float32. The function uses the default Euclidean
neighbors search. `get_params` and `set_params` use a typed
`LocallyLinearEmbeddingParams` record, with named updates also available.
Cloning starts a fresh estimator. Failed fits preserve the previous learned state.
Transform keeps the fitted output width until the next successful fit. It uses
the current `n_neighbors` and `reg` settings. `n_jobs` is forwarded to
NearestNeighbors, whose dense query currently runs serially.
`set_output` supports the default native array output.

Both `dense` and `auto` select the symmetric dense eigensolver at every sample
count. Sklearn's `auto` can select ARPACK above 200 samples. Explicit `arpack`
raises in Moxie. The dense solve is deterministic and independent of
`random_state`, `tol`, and `max_iter`. It uses the smallest algebraic eigenpairs,
skips the first, and orders the selected coordinates by absolute eigenvalue.
The reconstruction error is their signed sum. Eigenvector signs, and the
basis of any degenerate eigenspace, can differ across solvers. The live
neighbors engine orders distance ties by sample index.

## Numerical implementation

Standard LLE assembles the equivalent of `(I - W).T @ (I - W)` from the
local residuals. Modified LLE assembles multiple residuals per neighborhood
from the local spectrum, using the reference median energy ratio and
Householder adjustment. It recomputes local factors in the second pass to
bound temporary storage. Hessian LLE uses an unpivoted QR basis of the constant,
tangent, and quadratic terms. LTSA subtracts the constant and local tangent
projections from the identity. All four allocate a dense sample-by-sample
reconstruction operator and use a full dense eigendecomposition.

The local Gram products use NuMojo's validated rank-two `@`. The runtime
neighborhood size and sample count do not meet Numax's static-shape
`eigh`, `qr_factor`, `cholesky`, or `cholesky_solve` interfaces. The
covariance adapter uses mojo-linalg's Cholesky factors and solve, and
mojo-linalg supplies the runtime symmetric eigensolve and the local SVD
used when a neighborhood has more rows than features. The decomposition
owner's unpivoted `qr_basis` completes both the Hessian design and modified
LLE's left singular vectors to full bases via zero padding.
Column-pivoted QR would change the leading constant/tangent subspace used
by Hessian LLE.

## Validation

Run from the repository root with the locked Mojo environment on PATH.

```sh
python -u families/manifold/tools/check.py lle
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u families/manifold/tools/lle_parity.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u families/manifold/tools/lle_parity.py --notebook
```

Parity uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
It compares the estimator and function, reconstruction error, and held-out
coordinates for every method, including neighborhoods narrower and wider than
the input feature count. It checks a Hessian null-space degeneracy against the
sklearn reconstruction operator and checks a classifier pipeline. The parity script
aligns coordinate signs and allows larger tolerances for sklearn's Float32
functional fit and Float32 transform weights. The focused digits notebook
compares all four methods on the 200-sample subset.
