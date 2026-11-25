# Dense precomputed HDBSCAN hierarchy

The private `dense_brute_hierarchy(distances, min_samples, alpha=1)` builds
the merge tree for the dense precomputed brute path. It borrows a square
Float32 or Float64 NuMojo array with at least two samples, including
strided views. It widens to Float64 for arithmetic, matching sklearn's
estimator input promotion, and returns an owned `MergeTree` with ordered
Int64 children and Float64 merge heights. The caller's matrix stays intact
regardless of representation. Scikit-learn can instead mutate its
precomputed distance matrix when copying is disabled.

The kernel scales distances by `alpha` before selecting each row's
`min_samples`-th value, counting the diagonal. It uses the dense reference's
Prim traversal and edge orientation. Equal-distance merge ordering follows
the cluster family's existing NumPy 2.5.3 AVX2 argsort implementation. NaN
distances raise. Symmetric infinite distances represent missing edges, so a
returned merge height can be infinite.

The kernel uses linear scratch. Core distances use the standard library's
`sort` on a reused host row. Numax `partition` also performs a full sort and
copies through tensor storage, and NuMojo `sort_inplace` uses an unstable
quicksort. The standard library sort bounds the row work by O(n log n) without
packing the matrix or allocating a sorted matrix. The Prim scan takes O(n²)
time and agglomerative union-find assigns the merge IDs.

## Focused validation

From the repository root, with locked Mojo 1.0.0 / MAX 26.5.0 active:

```sh
python -B -c 'import sys; sys.path.insert(0, "families/cluster/tools"); import check; check.assemble(check.ROOT, check.ROOT / "DEPENDENCIES.json"); check.check_integration(); check.compile_one(check.ROOT, check.ROOT / "algorithms/hdbscan/tests/hierarchy_driver.mojo", "hdbscan_hierarchy_driver")'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B families/cluster/algorithms/hdbscan/tests/hierarchy_parity.py
```

The oracle compiles the pinned sklearn reachability and linkage Cython sources
using the locked sklearn installation for the supporting extension ABI. The
local `_hdbscan_brute` orchestration calls those kernels directly. Comparisons
require exact merge children and order and a relative tolerance of 2e-12 for
the heights read from the native driver's decimal output. Tests cover both
dtypes, strided input, ties, coincident samples, infinite distances, and invalid
shape and parameters. The case list and source hashes are recorded in
`families/cluster/builds/hdbscan-hierarchy-parity.json`.
