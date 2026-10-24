# Public API scaffold

Public names and callable parameter names/defaults follow the local scikit-learn
checkout. Newly declared constructors and operations raise `NotImplementedError`
through Mojo `Error`; they do not perform numerical work. Existing implementations
and family traits remain intact.

Mojo adaptations: heterogeneous inputs use `Some[Movable]`, accepting native
values without Python. Opaque configuration, learned attributes and deferred
results use the local `APIValue` marker. Numeric and dtype defaults use native values, including NaN and infinity.
Deferred Python callable/type defaults use `APIExpressionDefault(expression)`
to preserve their source identity; the exact reference signature is recorded
on every declaration. Python `**kwargs` becomes an optional named native value
parameter. Deferred operations raise; implemented estimators support the current Moxie
origin-preserving fluent mutation returns.
Properties are exposed as opaque fields. Inherited public methods are flattened
onto each struct; dynamic metadata-routing methods are outside this source-only
surface. These deferred declarations provide no numerical behavior or Python
reflection; the integrated implementations are described below.

`AffinityPropagation` and `affinity_propagation` are integrated: `EXPORTS.json`,
the family source imports and the live build manifests route both names to
`moxie.cluster._affinity_propagation`. Their former placeholder declarations
have been removed. See the [native API](../algorithms/affinity/docs/api.md).

Other algorithm names remain source declarations and are excluded from the
assembled runtime exports: `HDBSCAN`, `SpectralClustering`, `spectral_clustering`,
`SpectralBiclustering`, and `SpectralCoclustering` are deferred. The normal family build compiles the declaration
modules as well as the implemented runtime. `pixi run test` includes the existing
affinity/KMeans contracts and two isolated consumers that exchange fitted estimators;
`pixi run parity` checks both family-built drivers against local sklearn.

`KMeans`, `k_means`, and `kmeans_plusplus` now route to `moxie.cluster._kmeans`
in source exports and assembled manifests. Their placeholder declarations are
removed. See the [KMeans API](../algorithms/kmeans/docs/api.md) for supported
dense behavior, typed parameter/result records, and numerical limits.

`MiniBatchKMeans` routes to `moxie.cluster._mini_batch_kmeans`; its deferred
declaration is removed. See the [MiniBatchKMeans API](../algorithms/minibatch/docs/api.md).

`BisectingKMeans` routes to `moxie.cluster._bisect_k_means` in source exports
and assembled manifests. Its deferred declaration is removed. See the
[dense native API](../algorithms/bisecting/docs/api.md).

`DBSCAN` and `dbscan` route to `moxie.cluster._dbscan`; their deferred declarations
are removed. See the [dense native API](../algorithms/dbscan/docs/api.md).

`MeanShift`, `mean_shift`, `estimate_bandwidth`, and `get_bin_seeds` route to
`moxie.cluster._mean_shift`; their deferred declarations are removed. See the
[dense native API](../algorithms/mean_shift/docs/api.md) for dtype, RNG and numerical adaptations.

Dense `OPTICS` and its graph, DBSCAN-extraction and Xi-extraction helpers are
implemented in `algorithms/optics` and exported through the family and both
isolated consumers. See the [OPTICS API](../algorithms/optics/docs/api.md) for
metrics, typed sizes/results, Float64 promotion and unsupported capabilities.

`AgglomerativeClustering`, `FeatureAgglomeration`, `linkage_tree`, and `ward_tree`
route to `moxie.cluster._agglomerative`. Their deferred declarations are removed.
See the [dense native API](../algorithms/agglomerative/docs/api.md) for linkage,
tree cuts, feature pooling, typed configuration and unsupported options.

`Birch` routes to `moxie.cluster._birch`; its deferred declaration is removed.
See the [Birch API](../algorithms/birch/docs/api.md) for the CF tree,
integer/None global-clustering configuration, streaming behavior, and native
representations of tree handles and optional labels.
