# Neighbors contracts and dependency assembly

The public namespace follows `sklearn.neighbors`. The existing `_base` traits
are unchanged. New private plumbing is in `_contracts`; the `_distance` and
`_query_engine` native modules separate metric kernels and reusable search from
Python's larger `_base.py`. These are implemented native engines.
`TopKResult` and `RadiusResult` are native records, not sklearn public estimators.

## Owners and dependency closure

All paths below are relative to `moxie/families/neighbors`. Moxie owns the
family contracts and all listed internal projects.

| Project | Owned modules / exports | Additional dependencies |
| --- | --- | --- |
| shared/query_engine | `_distance`, `_query_engine`: private metric registry, metric evaluation, brute force, top-k selection, radius collection and shared weighting | none |
| shared/spatial_trees | `_binary_tree`, `_kd_tree` / KDTree, `_ball_tree` / BallTree, `_metrics` / VALID_METRICS | query_engine |
| algorithms/centroid | `_nearest_centroid` / NearestCentroid | query_engine |
| algorithms/nca | `_nca` / NeighborhoodComponentsAnalysis | query_engine |
| algorithms/nearest | `_unsupervised` / NearestNeighbors | spatial_trees (transitively query_engine) |
| algorithms/density | `_kde` / KernelDensity | spatial_trees |
| algorithms/knn | `_classification` / KNeighborsClassifier; `_regression` / KNeighborsRegressor | nearest |
| algorithms/radius | `_radius_classification` / RadiusNeighborsClassifier; `_radius_regression` / RadiusNeighborsRegressor | nearest |
| algorithms/lof | `_lof` / LocalOutlierFactor | nearest |

The radius private module split is deliberate: independent KNN/radius owners
cannot both contribute sklearn's single `_classification` or `_regression`
module. Their public names still export from `moxie.neighbors`. Each child adds
only its own symbols to the export manifest. NearestCentroid and NCA do not gain
query methods merely because they consume shared distance operations. LOF's
novelty-only prediction and density score summation remain estimator concerns.
Sparse exports in the original ownership map are deferred in [scope](scope.md).

## Numerical ownership and layout

Use pinned NuMojo 0.10.0, revision
515fb2856f0ecf3d2740a34d958fe168183e1129, Mojo 1.0.0 / MAX 26.5.0 on linux-64.
Read `../../../shared/numerics/docs/numerics.md` before choosing operations.
Input is `numojo.NDArray[DType.float32]` or `[DType.float64]`, with samples in
rows. Targets are estimator-owned shapes, never packed into a feature matrix.
Shape and indexing metadata are `numojo.Shape` and `Item`; strides are in elements
and access is offset-aware. Computation borrows input; `fit_snapshot` validates
then returns `.copy()`. Successful fit commits an independent snapshot; normal
refit replaces it, and failed validation must not publish partially fitted state.
Views explicitly alias and retain their allocation; arbitrary caller mutation
and concurrent access are not approved fitted-state semantics. Copying strided
views may copy the full backing buffer. Reshape, transpose and astype are not
promised zero-copy. This native ownership policy intentionally snapshots even
where sklearn may retain a compatible array without copying.

`validate_matrix(x, expected_columns=-1, precomputed=False, training=False,
allow_nan=False)` checks rank, nonempty dimensions, finite values and expected
columns. Dense precomputed fit is square; query columns equal n_samples_fit_;
values are nonnegative, with no imposed symmetry or zero-diagonal requirement.
`allow_nan` is reserved for nan_euclidean feature input, not precomputed input.
The query engine owns metric-specific missing-value distance semantics.

`TopKResult[dtype]` has `n_queries: Int`, `k: Int`, row-major
`indices: List[Int]` of length n_queries*k and
`distances: Optional[NDArray[dtype]]` of shape (n_queries,k).
`RadiusResult[dtype]` has `offsets: List[Int]`, `indices: List[Int]` and
`distances: Optional[NDArray[dtype]]` of shape (len(indices),).
Offsets length is n_queries+1, starts at 0, is nondecreasing and ends at
len(indices); row i occupies [offsets[i], offsets[i+1]). Empty rows are valid.
Indices are native signed 64-bit Int on the supported platform and always index
original fitted rows, never a tree's permuted position. Lists represent only
algorithmic index structure. Distances use the accepted numerical storage.

Records own all fields. Moves transfer buffers; `.copy()` deeply copies lists
and arrays. No references to an engine's scratch memory may escape. Optional
absence implements return_distance=False without fabricated zeros or a changed
method signature. `validate(n_samples_fit)` checks layout/index ranges; radius
also accepts `sorted_distances=False`. These checks allow positive infinite
output distances and reject negative distances. Top-k validation accepts an
explicit `allow_nan=True` for undefined nan_euclidean distances when rows have
no common observed features: NaNs must follow defined distances in each row,
matching the reference fallback sorting. Radius membership never includes an
undefined distance. Otherwise NaN distances are rejected. Order is checked when
required. They cannot certify nearestness,
duplicate-free membership, self-exclusion or radius membership; engine parity
must test those against input data. Records may be assembled incrementally by
engines but must pass validation before publication.

Numeric payload cost: top-k O(q*k) indices and optional distances; radius O(q+m)
metadata and optional m distances. Fit snapshots cost the copied backing storage.
No full q*n distance matrix is mandated. Engines may stream blocks; temporary
heaps are engine-owned and never replace these shared result records.

## Query and metric seams

`DenseMetric: Copyable, Movable` has associated `dtype`,
`validate_features(n_features: Int) raises`, and
`distance(x: NDArray[dtype], x_row: Int, y: NDArray[dtype], y_row: Int)
raises -> Float64`. Row bounds and validated features are caller obligations.
Metric configuration is immutable during a query; deep copies own parameter
arrays. `shared/query_engine` supplies typed parameter records for p, feature
weights, V/VI and native callable state, normalize sklearn aliases and retain
`effective_metric_`/`effective_metric_params_`. Native callable structs implement
this same trait rather than executing Python. Distance returns the actual metric;
reduced/squared distances are internal optimizations with an explicit monotone
conversion owned by the metric implementation. Tree bound calculations may use
that conversion, never redefine distance formulas. Precomputed input bypasses
metric evaluation. One registry at the engine owns metric meaning; the tree
owner publishes VALID_METRICS by backend capabilities, not a duplicate registry.

The registry covers the pinned dense brute metrics/aliases from
PAIRWISE_DISTANCE_FUNCTIONS and SCIPY_METRICS in `_base.py`, and the tree metric
classes in `_kd_tree.pyx.tp` / `_ball_tree.pyx.tp`. Weighted Minkowski, p<1 brute
semimetrics, callable support and missing-value policies require explicit
capability checks. Tree eligibility must reject nonmetrics or unsupported
weights/callables and precomputed modes, preserving the reference's backend
restrictions; 'auto' can select only an actually implemented eligible engine.
No fallback may silently change the chosen metric or requested backend.

`DenseQueryBackend: Movable` has associated `dtype` and:

- `fit(mut self, x: NDArray[dtype]) raises` (configuration is constructed separately).
- `kneighbors(self, x: Optional[NDArray[dtype]], k: Int, return_distance=True)
  raises -> TopKResult[dtype]`.
- `radius_neighbors(self, x: Optional[NDArray[dtype]], radius: Float64,
  return_distance=True, sort_results=False) raises -> RadiusResult[dtype]`.

`None` queries training rows and excludes each row's own index, preserving other
duplicate points. Explicit X, even when equal to fitted data, includes self.
`validate_k` requires k>0 and k<=n_fit for explicit X, k<n_fit for None; it raises
NotFittedError when n_fit<=0. Index availability is checked before allocation.
Top-k outputs are nondecreasing in actual distance. Equal-distance neighbor
selection/order is backend-dependent in sklearn; no global index tie-break is
promised. Engines must be deterministic for fixed input/configuration and tests
must allow equivalent tied selections while testing their downstream effects.
Radius inclusion is d<=radius. Unsorted output order is unspecified;
sort_results=True requires return_distance=True and nondecreasing distances.
Radius zero includes coincident points; positive infinity is allowed. Negative
or NaN radius is rejected by the shared typed options helper. Public adapters
must document this validation explicitly (reference method-level overrides are
less strict than constructor constraints); no numerical results are claimed here.

The existing public capability traits use one associated NeighborResult, so an
estimator implementing both queries must use an estimator-specific
`Variant[TopKResult[dtype], RadiusResult[dtype]]` (tested by the native adapter
probe), while the private backend returns concrete records. Typed configuration,
in-place fit, prefixed Mojo Error and unfitted clone behavior follow interfaces.md.
No universal probability, transformer or incremental capability is added.
KDE's tree density integration and tree inspection/count-only methods are separate
extensions owned by spatial_trees; they must not burden every query backend.
Public tree `query` / `query_radius` defaults and return ordering are adapter
contracts, distinct from estimator kneighbors/radius_neighbors defaults.

## Assembly and native compatibility

`DEPENDENCIES.json` (assembly schema 1) is a flattened live-source manifest:
each project declares `id`, `path`, `modules`, `assets` and `exports`, with no
revision or content-hash pins. It directly consumes current `../..` HEAD
(including working edits), its shared numerics, RNG, labels, linalg and special
projects, NuMojo, mojo-linalg, Numax and the shared assembly project.
`EXPORTS.json` explicitly owns family exports. Builds run the current
`../../shared/assembly/tools/consume_family.py` launcher on every build and
compile only `builds/deps`. No editable source fork or import-path-order
selection is allowed; a missing source checkout or file fails the build.

For a child, keep the upstream entries, add this family source plus every
transitive engine/algorithm actually used, owned modules and explicit package
export manifests. Paths resolve relative to that child's manifest. Use one moxie
namespace assembled under its own ignored builds/deps. Editing sources never
requires repinning; keep module/export declarations accurate when modules move.
Overlap of identical baseline source is deduplicated only under the shared
assembler's verified baseline rule; other ownership collisions fail. A relocation
requires an explicit reviewed ownership override, not a cloned helper module.
Keep compiler/MAX versions exact; a native incompatibility blocks consumption.
All numerical kernels remain pure Mojo; Python orchestrates builds/tests only.

`tests/consumers/a` and `b` have independent Pixi manifests/locks and generated
namespaces. Both compose ConsumerA and ConsumerB from distinct source projects,
call the same fit_snapshot, row_squared_norms and RandomState, and test
unfitted errors, owned snapshots and repeat-fit independence. They are test doubles.
Each can run `pixi run build` / `pixi run test`; the family tasks also compile and
execute both. Missing sources and module/export conflicts must fail before replacing a valid
namespace. Negative compilation proves sparse exports unavailable. The native
example `query_buffers.mojo` packs a known result without inventing a search.


## Current implementation assembly

The ownership map above remains authoritative. Family-wide integration now composes
`algorithms/nearest`, `algorithms/knn`, `algorithms/radius`, `algorithms/lof` and
`algorithms/density` with `shared/query_engine`. The KNN owner additionally owns
private `_supervised`, reused by radius target aggregation. Callable/builtin
weighting contracts reside at the shared query-engine owner; no common foundation
code is copied into it. Child modules are built in the family environment, not
through competing source include paths. Each owner's `EXPORTS.json` and module
entries are declared in the flattened manifests. The old raising `_api` and `_trees_api` inventories are excluded from assembly.

External dependencies are read live from their local checkouts on every build;
there are no cached snapshots or Git recovery fallbacks. See [estimators.md](estimators.md) for
concrete result-type and capability-trait adaptations.

NearestCentroid is composed from `algorithms/centroid` and reuses `_distance`,
shared column reductions and the KNN owner's existing `_supervised` target/score
helpers. It does not depend on spatial trees. Both downstream consumers fit
and score the real estimator.

## Spatial tree owner

`shared/spatial_trees` owns `_binary_tree` inspection/kernel helpers,
`_tree_engine` shared traversals, `_kd_tree` / `KDTree`, `_ball_tree` / `BallTree`
and `_metrics` / `VALID_METRICS`. Compile-time geometry specialization shares
selection, radius, density and dual-tree algorithms without duplicating engines.
Its export manifest participates in the family and both downstream consumers.
`_unsupervised` selects explicit or eligible auto trees; `_kde` uses their density
routines. Global capability traits are unchanged. See
[KDTree](kdtree.md) and [BallTree](balltree.md).

NCA is composed from `algorithms/nca`, exporting `_nca` and private
`_nca_math`, `_nca_lbfgs` and `_nca_linesearch`. It reuses existing SVD, RNG, dense reductions and target helpers;
its optimizer is local to the estimator. Both downstream consumers fit and
transform with the real estimator. See [NCA](nca.md).

Public tree structs delegate to SpatialTree, preserving inferred built-in
constructors while adding typed callable BallTree metrics. Search, geometry and
selection remain implemented once in the shared engine. ClassificationTargets
at the KNN owner adapts shared moxie._labels domains for all family classifiers
and NCA. Labels and RNG are consumed live from Moxie HEAD.
