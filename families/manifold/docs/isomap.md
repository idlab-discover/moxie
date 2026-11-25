# Dense Isomap

```mojo
from moxie.manifold import Isomap

var model = Isomap(n_neighbors=5, n_components=2)
var coordinates = model.fit_transform(X)
var projected = model.transform(held_out)
print(model.reconstruction_error())
```

Pass rank-two NuMojo arrays with rows in sample order. Float64 is the
default. Use `Isomap[DType.float32]` for Float32 input, geodesic distances,
coordinates and reconstruction error. Use `n_neighbors=None, radius=r` to
select radius neighborhoods. Training queries exclude the sample itself.
Held-out queries include every fitted sample as a candidate and raise
`ValueError` if a radius query finds no neighbors.

`metric="precomputed"` accepts a complete square training distance
matrix and query rows with one column per training sample. Distances must
be finite and nonnegative. Sparse graph input remains under the repository's
sparsity deferral. Other metric names and algorithms follow dense
`NearestNeighbors`. The optional `metric_params` is the neighbors engine's
typed `Metric[dtype]` record from `moxie.neighbors._distance`. It supplies the
effective metric and options when present, such as weighted Minkowski via
`Metric[DType.float64]("minkowski", p=3, w=weights.copy())`. Use
`get_params` and `set_params` to read and update `IsomapParams`, or use
keyword updates. The record can reset optional parameters to `None`.
Python callables, metadata routing and labeled output containers are
outside this native interface.

Training symmetrizes the neighbor graph by taking the minimum of both
directions. It keeps zero-weight edges. For disconnected graphs, Isomap
connects the nearest pair of samples between each pair of original
components and records sklearn's warning in `fit_warnings` using the shared
`moxie.warnings.FitWarning` contract. Neighbor-query ties
use the neighbors engine's deterministic ordering. Repair ties use training
row order. Floyd–Warshall (`"FW"`) and Dijkstra (`"D"`) accumulate shortest
paths in Float64. The default `"auto"` chooses Floyd–Warshall when the initial
directed neighborhood has at least one edge per four matrix entries, and
Dijkstra otherwise. Both algorithms scan dense rows and use quadratic
storage. Numax supplies no graph shortest-path primitive. The native loops
stream row updates and avoid broadcast pairwise intermediates.

Isomap fits decomposition's `KernelPCA` on the negative half squared
geodesic distances. The `"auto"` and `"arpack"` eigensolver settings use the
owner's dense exact implementation, which also handles `"dense"`. The owner
retains `tol` and `max_iter` as configuration. For each held-out row Isomap
takes the shortest path via any queried neighbor, builds its cross-kernel row
and projects with the fitted KernelPCA. The radius query API takes an optional
owned array, so radius transforms pass it a copy.

Fitting records `embedding_`, `dist_matrix_`, `nbrs_`, `kernel_pca_` and
`n_features_in_` on success. The arrays, neighbors estimator and KernelPCA are
optional before fitting. A refit replaces these snapshots, and cloning copies
configuration into an unfitted estimator. Parameter updates take effect at the
next fit. `fit_transform` returns an independent copy of the fitted embedding.
`get_feature_names_out` returns `isomap0`, `isomap1` and so on.

`reconstruction_error()` uses the Float64 Frobenius residual between the
centered geodesic kernel and the Gram matrix of the fitted coordinates,
divided by the training row count. This computes the same quantity as the
reference eigenvalue formula while avoiding cancellation from subtracting
nearly equal squared norms. The implementation reuses `KernelCenterer` and
fuses the Gram-entry calculation with the residual reduction to avoid a
second square array from NuMojo `@`. Small Float32 errors can differ from
sklearn's cancellation-sensitive score.

Run these focused commands from the repository root with the locked toolchain.

```sh
python -u families/manifold/tools/check.py isomap
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u families/manifold/tools/isomap_parity.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u families/manifold/tools/isomap_parity.py --notebook
```

Parity uses Isomap, KernelPCA and the graph repair helper from pinned sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. It checks both dtypes,
fixed-k and radius neighborhoods, all path settings, dense precomputed
distances, zero-weight edges, graph repair, held-out transforms and
FeatureUnion composition. Embedding comparisons align eigenvector signs.
Tolerances account for neighbor-distance rounding in Float32, distance
reduction order, the Jacobi eigensolver and fractional Minkowski powers.
Float32 reconstruction error is checked against the omitted spectrum of
the Float64-centered reference geodesic kernel.
