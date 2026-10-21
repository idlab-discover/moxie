# KDTree

`moxie.neighbors.KDTree[dtype]` is an owned, pure Mojo KDTree for dense
Float32/Float64 matrices. Its implementation lives in `shared/spatial_trees`.
It partitions original row indices at the median of the widest coordinate,
stores a complete binary tree with tight axis-aligned bounds, and prunes search
using the selected metric. Numerical storage uses NuMojo; point distances reuse
the family's metric implementation.

```mojo
from moxie.neighbors import KDTree

var tree = KDTree(X, leaf_size=40, metric="euclidean")
var nearest = tree.query(Q, k=5)
var within = tree.query_radius(Q, 0.5, return_distance=True)
var counts = tree.query_radius(Q, 0.5, count_only=True)
var density = tree.kernel_density(Q, h=0.2, return_log=True)
```

Construction accepts `data`, `leaf_size=40`, `metric="minkowski"`, optional
`sample_weight`, keyword `p=2`, and optional typed `metric_params: Metric[dtype]`.
The latter replaces metric configuration, as in the existing estimators. Valid
names are `euclidean`, `l2`, `minkowski`, `p`, `manhattan`, `cityblock`, `l1`,
`chebyshev`, and `infinity`; `KDTree.valid_metrics` supports length/indexing.
Minkowski requires p >= 1 under the family eligibility contract (the reference
standalone constructor also accepts p<1). Weighted feature metrics, callable metrics and
precomputed distances raise explicitly. Sample weights are separate from feature
weights: finite nonnegative sample weights affect density only. Zero total
sample weight produces zero density (negative infinity in log space).

Methods follow the reference names/defaults:

- `query(X, k=1, return_distance=True, dualtree=False, breadth_first=False,
  sort_results=True)` returns `TopKResult[dtype]`, with flat original-row indices
  and optional `(n_queries,k)` distances. Both single-tree and dual-tree pruning
  are implemented. Breadth-first uses a node queue; depth-first uses a stack.
  Sorted output is also returned when sorting is not requested, which permits
  unspecified ordering. Ties are deterministic by original index and can differ
  from sklearn's unspecified tie selections.
- `query_radius(X, r, return_distance=False, count_only=False,
  sort_results=False)` accepts a scalar Float64 or rank-one Float64 array of
  length one or n_queries. Its typed `TreeRadiusResult` contains exactly one
  optional `neighbors: RadiusResult[dtype]` or `counts: List[Int]`. Membership is
  inclusive. Negative radii produce empty rows; NaN raises; infinity includes
  all points. Count-only avoids allocating membership/distance output. Conflicting
  count/distance options and sorting without distances raise.
- `kernel_density(X, h, kernel="gaussian", atol=0, rtol=1e-8,
  breadth_first=True, return_log=False)` supports Gaussian, tophat,
  Epanechnikov, exponential, linear and cosine kernels. It returns an
  **unnormalized weighted sum**, shape `(n_queries,)`, retaining tree dtype.
  All supported metrics can be used; probability-density normalization is valid
  only for Euclidean distance, as in sklearn. Exact log-domain summation meets
  nonnegative error tolerances without approximation. Compact-support kernels
  prune disjoint boxes; infinite-support kernels visit all leaves.
- `two_point_correlation(X, r, dualtree=False)` returns a `List[Int]` of
  cumulative pair counts for each supplied rank-one Float64 radius, retaining
  input order, including unsorted and repeated radii. It counts ordered pairs,
  including self pairs when X is the training set. Both traversal modes exist.
- `get_arrays()` returns a `TreeArrays` record containing independent copies of
  `data`, `idx_array`, `node_data`, and `node_bounds`. Node records have
  `idx_start`, `idx_end`, `is_leaf`, and `radius`; bounds have shape
  `(2,n_nodes,n_features)`. Inspectable public fields use the same layout.
- `get_n_calls()`, `reset_n_calls()`, and `get_tree_stats()` report actual point
  distance evaluations and cumulative `(trims, leaves, splits)` from top-k
  traversals. Construction and box-bound arithmetic do not count as point
  distances. Reset clears only distance calls. Exact values depend on traversal
  and are not promised identical to sklearn. The pinned sklearn revision also
  accumulates stats; its Python deprecation warnings are not emitted in Mojo.

All public query methods borrow arrays and mutate only their tree's counters.
Concurrent access to one mutable tree is not supported. `get_arrays()` copies
rather than exposing mutable backing views. Direct mutation of inspectable tree
fields is unsupported. Input matrices are rank two and nonempty; general leading
batch axes, Python object arrays, pickling/joblib, reflection and Python metric
objects are not represented. These language adaptations are tested explicitly.
Float32 point arithmetic widens through the family metric, then casts output.
Minkowski inherits that metric's documented arbitrary-power numerical limits.

`NearestNeighbors` and its KNN/radius/LOF consumers accept `algorithm="kd_tree"`.
`auto` selects KDTree for eligible metrics, at most 15 features and fewer than
half as many requested neighbors as fitted samples; eligible BallTree metrics
then select BallTree, and other cases use brute force.
An estimator's implicit training query excludes each row's own index but retains
other duplicates. A public tree query always includes self. Failed fits preserve
previous fitted state. KDE supports `auto`, `kd_tree` and `ball_tree`, exposes an optional
fitted tree variant `tree_`, and uses tree density with its normalization and learned bandwidth.
KDE accepts tree-supported metrics; only Euclidean normalization is a probability
density. Its seeded sampling contract remains.

Construction uses deterministic index merge sorting: O(nd log n + n log² n)
time and O(nd) bounds/data plus O(n) index scratch. Top-k insertion selection
costs O(k) per accepted candidate; no performance equivalence to sklearn's heaps
is claimed. All-pairs work remains possible on poorly pruning data. No benchmark
claim is made.

## Validation

Run `pixi run test` and `pixi run parity`. Native contracts cover dtype, options,
self exclusion, counters, ownership, invalid inputs and inspection. The parity
bridge sends fixtures to a compiled Mojo executable and compares queries,
radius memberships, correlations, geometry and six weighted kernels with SciPy.
The validated run covers 336 independent configurations and 205 original sklearn
test invocations, including its Float32/Float64 consistency tests. It uses rtol 3e-5 for Float32 and 2e-11 for Float64 query fixtures; kernel tests allow 2e-9, a conservative bound since the
kernel sums moved to Numax fdlibm exp/log. Log tails
use corresponding absolute tolerance. Membership/count comparisons are exact.

For the original sklearn test bodies, compile the recorded local checkout into
a separate Python test environment (Python, Cython, Meson, Ninja, NumPy, SciPy,
pytest, joblib, threadpoolctl and the checkout's declared runtime dependencies).
Keep builds under ignored `builds/`; never modify the external reference.
Set `SKLEARN_ORACLE_PYTHON` to that environment's Python when running
`pixi run parity`. `tests/parity_kdtree.py` imports test bodies directly from
`../../../../scikit-learn/sklearn/neighbors/tests/test_neighbors_tree.py` and invokes query,
radius, correlation, kernel-density and SciPy Gaussian-density tests through a
native result adapter. Python-only serialization, object input and joblib tests
are outside the Mojo contract. It also checks the reference Git revision and
source cleanliness. A missing compiled sklearn is reported explicitly; independent
parity still runs, but does not count as original compiled-oracle coverage.

See [BallTree and VALID_METRICS](balltree.md). NCA remains separate work.
Sparse graph APIs remain deferred.
