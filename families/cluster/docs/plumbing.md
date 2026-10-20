# Dense clustering plumbing

The root supplies contracts and validation and integrates the dense affinity
implementation from `algorithms/affinity` and KMeans from `algorithms/kmeans`
into its public runtime exports, together with MiniBatchKMeans from
`algorithms/minibatch`, BisectingKMeans from `algorithms/bisecting`, and DBSCAN
from `algorithms/dbscan`, plus MeanShift and its bandwidth/bin-seed helpers
from `algorithms/mean_shift`. MeanShift reuses live radius and nearest queries
for shifting, merging and labeling. DBSCAN consumes the live neighbor `RadiusResult`
offsets/indices directly, avoiding conversion to the older `Neighborhoods`
record. This is a dense ragged index representation, not a sparse matrix.
The existing traits and public imports in `moxie.cluster._base` are unchanged.
The behavioral reference is sklearn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The module split follows sklearn's `_kmeans`, `_agglomerative`, `_dbscan`,
`_optics`, `_hdbscan`, `_mean_shift`, `_birch`, `_spectral`, `_bicluster`,
`_affinity_propagation` and `_bisect_k_means` boundaries. Shared native records
live in private `_contracts`; Python tuple/array reflection is not required.

## Storage and exchange interfaces

Import `NDArray`, `Shape`, `Item` from current NuMojo. Real arrays use compile-time
Float32 or Float64; indices/labels use `NDArray[DType.int64]` (`Labels`).
X is `(n_samples,n_features)`, centers `(n_centers,n_features)`, labels
`(n_samples,)`, weights `(n_samples,)`. Kernels borrow inputs; returned results
own their buffers. `snapshot_centers(x)` validates nonempty rank 2 and deep
copies. It preserves NuMojo layout, including copying the backing allocation
for strided views. `.view()` explicitly shares storage and lifetime; `.copy()`
is independent. No implicit conversion or runtime dtype promotion is promised.
Strides count elements, offsets are handled by NuMojo; positive/negative strides
and offsets retain the numerical owner's contract. Contiguous materialization
must be explicit and costed by the consuming engine. A fit snapshot must not
retain a caller's mutable view. Clone copies constructor configuration only;
normal refit replaces learned state after successful validation. Reuse foundation
`FittedState` for ordinary tabular lifecycle checks.

`validate_matrix(x)` checks only rank/nonempty shape, deliberately not finiteness:
HDBSCAN handles nonfinite rows itself. `validate_centers(x, centers)` additionally
checks feature agreement; it does not impose n_centers <= n_samples on prediction.
`validate_labels(labels,n_samples,n_clusters,minimum=0)` checks rank, length,
and bounds. Minimum is exactly 0 (ordinary), -1 (noise/failure labels), or -3
(HDBSCAN: -1 noise, -2 infinity, -3 missing). Cluster IDs remain unsorted in sample
order, with no renumbering. All-noise output allows n_clusters=0. Individual
algorithms validate which labels actually occur and preserve reference ordering.
Errors are Mojo `Error` with a `ValueError` prefix.

`CentroidResult[dtype]` owns `labels`, `centers`, Float64 `inertia`, and Int
`n_iter`. This is an internal exchange record, not a public k_means return ABI:
the estimator/helper maps optional public outputs explicitly. Inertia means
sum of sample-weighted squared distances; score is its negative where the
reference specifies that. Assignment ties, zero-weight handling, relocation,
Lloyd/Elkan scratch, convergence, initialization and incremental updates belong
to centroid_engine/KMeans sessions, not scaffold helpers. Center updates consume
X, sample weights and assignments in sample order and produce the same dtype
centers and cluster masses; an engine must not add a second storage container.
Engine-specific scratch and modes may extend the private engine interface;
changes to these shared exchange records require a reviewed scaffold revision.

`MergeTree` owns Int64 `children` `(n_merges,2)`, Int `n_leaves`, Int
`n_connected_components`, optional Int64 `parents` `(n_leaves+n_merges,)`, and
optional Float64 `distances` `(n_merges,)`. Leaves are 0..n_leaves-1; row i creates
node n_leaves+i. Child IDs precede their parent and cannot be consumed twice.
Roots parent themselves when parents are supplied. `validate()` checks these
invariants and optional shapes. It permits partial forests for extension;
dense unstructured linkage currently requires the child engine to produce a
complete tree and parents=None, matching sklearn. Distances=None means not
requested, never a zero-filled placeholder. The linkage owner checks distance
semantics and cut rules. HDBSCAN's condensed tree is a separate algorithm-specific
structure; it must not be forced into this binary-tree record.

`Neighborhoods` owns `List[List[Int]] indices` and Int `n_reference`. Row q is
query q's list of reference indices; empty rows are valid. `validate(n_queries)`
checks row count and bounds. This is a ragged algorithmic index structure, not
CSR. The query owner supplies radius inclusion (<=), duplicate-sample handling,
self inclusion for DBSCAN, and ordering according to the reference. No sorting
or deduplication is silently performed. Distance-bearing queries may add paired
NuMojo per-row arrays via a separately reviewed adapter; this record is the
index-only DBSCAN/mean-shift boundary.

`DenseDistanceAdapter` declares `dtype` and
`pairwise(self, x: NDArray[dtype], y: NDArray[dtype]) raises -> NDArray[dtype]`.
Output is `(x.rows,y.rows)` in input row order. The concrete adapter explicitly
states metric and squared-versus-unsquared units. Dense precomputed distances
remain dense and must be validated by the estimator's metric-specific rules;
HDBSCAN's permitted infinity cannot be globally rejected.
`RadiusQueryAdapter` declares `dtype` and
`radius_neighbors(self,x: NDArray[dtype],radius: Float64) raises -> Neighborhoods`.
The fitted reference set is owned by the implementing neighbor adapter. These
traits contain no fallback engine. Raising test doubles establish signatures only.

RNG is the shared `moxie._random.RandomState`, not a cluster wrapper. Integer
configuration creates a fresh stream at the reference reset point. Explicit
mutable streams are borrowed and advanced; None is the foundation's retained
application context. Preserve draw order and compile with contraction disabled.
The foundation RNG docs define accepted sampling operations and exact limitations.

## Owners and deferred implementation sessions

Every row below depends on `cluster.scaffold`; additional dependencies are listed.
Paths identify internal projects; scope.md distinguishes implemented and deferred work.

| Child path | Public exports / owned capability | Additional dependencies |
| --- | --- | --- |
| shared/centroid_engine | Private centroid assignment/update | none |
| algorithms/affinity | AffinityPropagation, affinity_propagation | none |
| algorithms/kmeans | KMeans, k_means, kmeans_plusplus | cluster.centroid_engine |
| algorithms/minibatch | MiniBatchKMeans | cluster.kmeans |
| algorithms/bisecting | BisectingKMeans | cluster.kmeans |
| algorithms/bicluster | SpectralBiclustering, SpectralCoclustering | cluster.kmeans, decomposition.svd_engine |
| algorithms/agglomerative | AgglomerativeClustering, FeatureAgglomeration, linkage_tree, ward_tree | neighbors.nearest |
| algorithms/dbscan | DBSCAN, dbscan | neighbors.nearest |
| algorithms/optics | OPTICS, compute_optics_graph, cluster_optics_dbscan, cluster_optics_xi | neighbors.nearest |
| algorithms/hdbscan | HDBSCAN | neighbors.nearest |
| algorithms/mean_shift | MeanShift, mean_shift, estimate_bandwidth, get_bin_seeds | neighbors.nearest |
| algorithms/birch | Birch | cluster.agglomerative |
| algorithms/spectral | SpectralClustering, spectral_clustering | manifold.spectral, cluster.kmeans |

KMeans and MiniBatchKMeans share sklearn's `_kmeans` source boundary. Native
contributions must have distinct implementation modules (e.g. `_kmeans` and
`_mini_batch_kmeans`) and merge only public package exports through manifests.
This avoids two source owners for one module. `_BaseKMeans`, `BaseSpectral` and
`AgglomerationTransform` are private source classes, not promised public exports.
Bicluster memberships are boolean arrays `(n_biclusters,n_rows/n_columns)`;
get_indices returns separate Int64 vectors, get_shape an Int pair, get_submatrix
an owned dense selection. Bicluster engine/estimator contracts remain their own
session. No prediction, weighting, transformation or incremental capability is
imposed on all clusterers. FeatureAgglomeration must preserve its reference
method availability rather than inherit an operational fit_predict accidentally.

## Assembly and compatibility

`DEPENDENCIES.json` declares live foundation, numerics, NuMojo, random and
family source paths, module ownership, exports and license assets. Builds use
current Moxie HEAD and working sources. The root build uses Moxie's complete manifest; isolated consumers use their
local manifests. `tools/check.py` imports Moxie's current
namespace builder directly; no pinned launcher, source hashes, repinning scripts
or historical dependency fallback is used. The generated namespace lives under
ignored `builds/deps`. Receipts describe consumed sources, without gating them.
Missing sources and module/export collisions fail before replacing a namespace.

Both `tests/consumers/a` and `b` combine both independent contribution modules
and run cross-module buffer/RNG calls against current shared code. Each has a
standalone Pixi build/test entry point. Tests check ownership, copy/view behavior,
live source updates, missing sources and module/export conflicts. Preserve the
Mojo 1.0.0 / max-core 26.5.0 toolchain lock. Runtime execution remains pure Mojo;
Python, NumPy and SciPy are validation/tooling dependencies only.

Public wrappers must preserve sklearn constructor defaults and parameters,
learned attributes ending in `_`, optional return branches, score semantics and
method availability. Typed parameter records and origin-preserving fluent mutation returns follow
current Moxie contracts. In particular DBSCAN permits negative sample weights;
shared weight validation must not impose positivity across the whole family.
Numerical finiteness, metric properties, tolerance policies and weights are
estimator-specific checks, not claims of the structural validators here.

The centroid seam is typed independently of the distance/query seams:
`CentroidAssignment.assign(self,x,centers,weights) -> AssignmentResult` and
`CentroidUpdate.update(self,x,labels,weights,previous) -> CenterUpdate[dtype]`
(both raising; real arrays have the adapter's compile-time `dtype`).
`AssignmentResult` owns Labels and Float64 inertia. `CenterUpdate` owns centers
`(k,d)` and masses `(k,)` in the input dtype. The prior centers fix k/d and permit
an engine-specific empty-cluster policy. Inputs are immutable borrows; scratch,
relocation and streaming accumulators are engine-owned. The centroid owner
implements these seams or composes them inside Lloyd/Elkan specializations;
these traits do not prescribe a universal iterative engine. Explicit weights
are always passed at this seam; public optional weights are normalized by the
estimator. There is no default implementation or silently substituted strategy.

Dense `OPTICS` and its graph, DBSCAN-extraction and Xi-extraction helpers are
implemented in `algorithms/optics` and exported through the family and both
isolated consumers. See the [OPTICS API](../algorithms/optics/docs/api.md) for
metrics, typed sizes/results, Float64 promotion and unsupported capabilities.

Dense agglomerative clustering and feature agglomeration are implemented in
`algorithms/agglomerative`, with the `linkage_tree` and `ward_tree` helpers.
See the [agglomerative API](../algorithms/agglomerative/docs/api.md) for native
configuration, tree outputs, pooling, numerical contracts and scope limits.

Dense `Birch` is implemented in `algorithms/birch` and exported by the family
and both isolated consumers. It reuses agglomerative Ward clustering, the
centroid distance kernel, and neighbor queries. See its [API](../algorithms/birch/docs/api.md)
for CF-tree handles, optional labels, incremental behavior, and supported
`n_clusters` configurations. `pixi run parity` writes its comparisons to
`builds/birch-parity.json`.
