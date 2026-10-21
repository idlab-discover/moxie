# BallTree and metric discovery

`moxie.neighbors.BallTree[dtype]` indexes dense NuMojo Float32/Float64 matrices.
It partitions original row indices at the median of the widest coordinate,
stores a centroid and enclosing metric radius per node, and reuses the spatial
family's traversal engine. Inputs and metric parameters are independently copied.

```mojo
from moxie.neighbors import BallTree, NearestNeighbors, VALID_METRICS

var tree = BallTree(X, leaf_size=40, metric="canberra")
var nearest = tree.query(Q, k=5)
var counts = tree.query_radius(Q, 0.5, count_only=True)
var names = VALID_METRICS["ball_tree"]
var model = NearestNeighbors(5, algorithm="ball_tree", metric="canberra")
model.fit(X)
```

The constructor, query methods, result records, density kernels, correlation,
inspection and counter methods have the same defaults and typed contracts as
[KDTree](kdtree.md). Both single-tree and dual-tree queries are implemented.
`node_bounds` has shape `(1,n_nodes,n_features)` and stores centroids;
`node_data.radius` encloses each node's descendants. Sample weights affect
centroids and density sums. A node with zero total weight uses an unweighted
centroid; zero total tree weight gives zero density. This avoids undefined
centroids without changing query distances or weighted kernel sums.

`BallTree.valid_metrics` supports length and integer indexing. `VALID_METRICS`
is a read-only mapping-like constant with `len`, `keys()`, membership and string
indexing for `"ball_tree"`, `"kd_tree"` and `"brute"`. Indexing returns an owned
`List[String]`; changing it does not change the registry. Unknown keys raise.
The registry lists implemented names; parameter restrictions still apply.

BallTree supports the nine KDTree names/aliases plus `seuclidean`, `mahalanobis`,
`hamming`, `canberra`, `braycurtis`, `jaccard`, `dice`, `rogerstanimoto`,
`russellrao`, `sokalmichener`, `sokalsneath` and `haversine`.
Use `metric_params=Metric[dtype](...)` for feature weights, p, V or VI:

- Minkowski requires p >= 1 and accepts nonnegative feature weights, including
  zero weights. Weighted Minkowski requires finite p, as in the reference;
  unweighted p=infinity resolves to Chebyshev. KDTree continues to reject weighted Minkowski.
- Standardized Euclidean requires explicit V; omitted V inference belongs to
  brute self queries.
- Mahalanobis requires explicit VI or covariance V. V is converted through the
  shared covariance-inverse owner, which requires exactly symmetric positive
  definite V; unsupported/singular covariance raises.
- Haversine requires two coordinates in radians.
- Callable metrics use `BallTree[dtype, M](X, metric_params=callback)` with a
  DenseMetric implementation. The registry includes `pyfunc` as the reference
  callable capability name; Python callbacks are not executed. Callable bounds
  do not prune, preserving correctness for nonmetrics.
  Precomputed distances, cosine, correlation and p<1 belong to brute force.
  Legacy `wminkowski` remains a private brute-engine extension, not a BallTree
  or public-registry name.

Triangle bounds prune Euclidean/Minkowski, standardized Euclidean, Manhattan,
Chebyshev, Canberra, Hamming and Haversine queries. Other supported formulas
visit leaves without geometric pruning: Boolean centroid formulas, nonmetrics,
and arbitrary Mahalanobis VI must not silently lose candidates. Nonfinite bounds
also disable pruning. Bounds use a conservative floating-point guard; this is
not a certified interval-arithmetic implementation. Undefined or negative point
distances raise a clear ValueError, including all-zero Dice/Sokal-Sneath cases,
instead of returning incomplete neighbor results. These are explicit limitations
of the supported numerical domain.

Point distances retain the family's arithmetic and inclusive radius boundaries;
the local mojo-datastructures reference's outward-rounded distance is not used.
Float32 arithmetic widens before output conversion. Ties use original row IDs.
Arbitrary Minkowski powers retain the shared metric's numerical limitations.
Counters count point comparisons, excluding centroid/bound calculations; their
values need not equal sklearn's counters. Construction and queries have the same
worst-case costs as the documented KDTree implementation. No speed claim is made.

NearestNeighbors and its KNN/radius/LOF consumers support explicit `ball_tree`.
Auto selection prefers KDTree, then eligible BallTree, when feature count <= 15
and requested neighbors < half the fitted rows; otherwise it uses brute force.
KDE auto selection similarly prefers an eligible KDTree, then BallTree, without
the neighbor-count heuristic. KDE's `tree_` is an optional typed variant;
`metric_params` accepts the same complete Metric record. Non-Euclidean kernels
match metric-distance sums, but Euclidean normalization does not make those sums
normalized probability densities.

Run `pixi run test` and `pixi run parity`. The latter requires a compiled pinned
sklearn oracle for BallTree; set `SKLEARN_ORACLE_PYTHON` and, if installed into a
local target directory, `PYTHONPATH`. The selected Python runs all parity scripts,
so its NumPy/SciPy and compiled sklearn must be compatible. For example:

```sh
PYTHONPATH=/path/to/oracle-site-packages \
SKLEARN_ORACLE_PYTHON=/path/to/oracle-python pixi run parity
```

See the build instructions in [KDTree](kdtree.md).
The BallTree suite covers 602 independent configurations and 238 original pinned
sklearn test invocations. Tests cover both dtypes, all advertised metric families, weighted Minkowski,
traversal modes, node geometry, inclusive boundaries, duplicates, ownership,
invalid options, refits/cloning, KDE and two independently assembled consumers.
Query tolerances are 3e-5 (Float32) and 2e-11 (Float64); kernel/log-density
comparisons allow 2e-9 (conservative since the kernel sums moved to Numax fdlibm exp/log). Membership and pair
counts are exact. Original pinned sklearn test bodies run through a typed native
adapter; this does not establish full sklearn API or estimator-check parity.
