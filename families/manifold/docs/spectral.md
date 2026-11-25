# Dense spectral embedding

```mojo
from moxie.manifold import SpectralEmbedding, spectral_embedding

var model = SpectralEmbedding(2, affinity="rbf", eigen_solver="dense")
var coordinates = model.fit_transform(X)
var clustering_coordinates = spectral_embedding(
    affinity, n_components=3, eigen_solver="dense", drop_first=False,
)
```

Both entry points accept rank-two NuMojo Float64 and Float32 arrays, including
strided views, and return owned coordinates in the input dtype and sample order.
Use `SpectralEmbedding[DType.float32]` for Float32 input. `affinity="rbf"`
builds the kernel with `gamma` defaulting to the reciprocal feature count.
`affinity="precomputed"` and the standalone function accept square, finite,
nonnegative affinity weights. The diagonal is excluded from graph degrees.
The Laplacian uses the symmetric average of both input triangles. The estimator
preserves the original values in its owned `affinity_matrix_` snapshot and
records warnings in `fit_warnings` for material asymmetry and disconnected
graphs in that order. The standalone function returns an array and has no
warning record. Disconnected components and repeated eigenvalues can produce
different valid orthonormal bases across solvers. The normalized dense
Laplacian gives isolated vertices diagonal zero and degree scale one, matching
the reference's dense LAPACK branch.

The function supports both normalized and unnormalized Laplacians and
`drop_first=False` for consumers that need the first eigenvector, such as
spectral clustering. The estimator requires at least two samples and both
entry points require positive `n_components`. The dense solver returns up to
the number of available modes, including all eigenpairs with `drop_first=False`.
The standalone function returns zero columns when it drops a singleton's only
mode. In normalized mode the embedding divides each
eigenvector by the square root of each vertex's degree, then applies sklearn's
sign convention using the first largest-magnitude coordinate of each column.

Choose `eigen_solver="dense"`. The sklearn solver default `None` selects
ARPACK and raises here, as do explicit `"arpack"`, `"lobpcg"` and `"amg"`.
`eigen_tol=None` represents sklearn's `"auto"`; numeric tolerances and integer
seeds are validated, and the deterministic dense solver uses its own fixed
convergence criterion without an initial random vector. The default
`affinity="nearest_neighbors"` and `"precomputed_nearest_neighbors"` raise under
the existing sparse-graph deferral. The native affinity parameter takes named
dense modes. The sklearn callable extension remains open. The typed
`n_neighbors` and `n_jobs` configuration is retained for compatibility with the
reference constructor. Neither alters the dense affinities.

`embedding_`, `affinity_matrix_`, and `gamma_` use `Optional`, with `gamma_`
present only after an RBF fit. `n_features_in_` records the input column count.
A successful fit replaces learned state, and a failed fit preserves it.
`get_params` returns a `SpectralEmbeddingParams` record, `set_params` accepts the
record or keyword updates, and `clone` copies only the configuration. Set
optional fields to `None` through the record to reset them. `fit` and
`set_params` return a mutable borrow, and `fit_transform` returns an independent
copy of the learned coordinates. Precomputed affinities advertise pairwise
input. Dense arrays carry no column names, and metadata routing is a Python
binding capability.

The implementation reuses the kernel-ridge owner's `pairwise` RBF kernel,
shared numerics' `sum_axis2d`, and the manifold `DenseSpectralProblem` and
`DenseSpectralSolver` seam with mojo-linalg's ascending `symmetric_eigh`.
The Numax `linalg.eigen.eigh` API requires a compile-time matrix order, as does
`linalg.array.eigh`, whose fixed sweeps and unsorted result are also unsuitable
for requesting ordered modes of a runtime-sized input. The Laplacian
construction uses one symmetric Float64 work array and linear degree and
connectivity storage. The eigensolver uses quadratic storage and dense cyclic
Jacobi; the dense path computes the full eigensystem before selecting modes.

Validation uses the pinned sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Tests compare signed coordinates
for distinct eigenvalues, and compare projection matrices for repeated
eigenspaces. Independent Laplacian residual and orthogonality checks cover
both dtypes. Coordinate absolute and relative tolerances are `2e-10` for
Float64 and `3e-5` for Float32, whose reference builds the Laplacian and solves
at input precision. Affinity comparisons are stricter at `2e-13` and `3e-7`
respectively. Both normalized and unnormalized paths, the first-mode switch,
isolates, all eigenpairs, RBF and precomputed affinities, and digits RBF are
covered.

```sh
python -u families/manifold/tools/check.py spectral
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u families/manifold/tools/spectral_parity.py
python -u families/manifold/tools/spectral_parity.py --notebook
```

The Laplacian eigenmap formulation follows the reference implementation and
Belkin and Niyogi's paper. [Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](https://doi.org/10.1162/089976603321780317)
