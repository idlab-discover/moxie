# Dense HDBSCAN hierarchies

## Precomputed distances

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

## Finite feature inputs

The private `dense_feature_hierarchy(X, min_samples, metric, *, alpha=1,
algorithm="auto", leaf_size=40)` takes the neighbors family's typed
`Metric[DType.float64]` and a finite Float32 or Float64 NuMojo feature matrix.
The kernel borrows contiguous Float64 data, and packs Float32 or strided
data into a contiguous Float64 buffer. It returns an owned `MergeTree`.
NaN and infinite feature rows raise at this kernel boundary. The estimator
will handle filtering and remapping before and after this call.

- `brute` computes pairwise distances on demand. Both the core distances and
  pair distances use the same alpha scaling, and the diagonal contributes
  zero to each row's neighborhood. The Prim traversal uses the precomputed
  brute path's edge convention.
- `kd_tree` and `ball_tree` query explicit training rows through
  `NearestNeighbors.kneighbors`, so the queries include the sample itself.
  Core distances stay in the original metric and Prim scales the pair
  distances by alpha. Each Prim candidate retains the source that set its
  best reachability. The tree path raises on nonfinite distances.
- `auto` chooses KDTree for eligible metrics, then BallTree, then brute
  using the neighbors family's eligibility rules.

The brute path evaluates shared neighbor metrics in a row sort pass and in
Prim, avoiding an n-by-n pairwise matrix. Bray–Curtis uses the two distance
formulas from the pinned brute and tree backends. The brute denominator sums
`abs(x + y)` as SciPy does, while the BallTree metric sums
`abs(x) + abs(y)`. Tree queries use batches of up to 16 rows to bound the
returned top-k storage even when `min_samples=n_samples`.

## Focused validation

From the repository root, with locked Mojo 1.0.0 / MAX 26.5.0 active:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -c 'import sys; sys.path.insert(0, "families/cluster/tools"); import check; check.assemble(check.ROOT, check.ROOT / "DEPENDENCIES.json"); check.check_integration(); [check.compile_one(check.ROOT, check.ROOT / f"algorithms/hdbscan/tests/{kind}_driver.mojo", f"hdbscan_{kind}_driver") for kind in ("feature", "hierarchy", "tree")]'
python -B families/cluster/algorithms/hdbscan/tests/feature_parity.py
python -B families/cluster/algorithms/hdbscan/tests/hierarchy_parity.py
python -B families/cluster/algorithms/hdbscan/tests/tree_parity.py
```

The oracle compiles the pinned sklearn reachability and linkage Cython sources
using the locked sklearn installation for the supporting extension ABI. The
local `_hdbscan_brute` orchestration calls those kernels directly. Comparisons
require exact merge children and order and a relative tolerance of 2e-12 for
the heights read from the native driver's decimal output. Tests cover both
dtypes, strided input, ties, coincident samples, infinite distances, and invalid
shape and parameters. The case list and source hashes are recorded in
`families/cluster/builds/hdbscan-hierarchy-parity.json`.

Feature parity runs the original `_hdbscan_brute` and `_hdbscan_prims`
orchestration using the private linkage oracle and locked sklearn neighbor
and metric extensions. The feature driver also passes the hierarchy
through EOM or leaf extraction and compares labels and probabilities with
the pinned Cython tree oracle. Comparisons check exact merge children,
merge order, and labels. Heights and probabilities use 2e-12 relative
tolerance. Cosine additionally allows 2e-15 absolute error in heights and
1e-10 relative error in probabilities because subtracting the similarity
from one amplifies roundoff at small merge heights. See
`families/cluster/builds/hdbscan-feature-parity.json` for fixtures,
source hashes, and versions.
