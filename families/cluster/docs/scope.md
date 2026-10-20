# Dense family scope

Supported now: inherited cluster capability traits; Float32/Float64 dense shape
and center validation, independent center snapshots, Int64 label range policies,
centroid result ownership, binary merge-tree structural validation, ragged
neighbor index validation, and typed dense distance/radius-query extension points.
The family build also exports the implemented dense `AffinityPropagation` and
`affinity_propagation` from `algorithms/affinity`. See its
[scope](../algorithms/affinity/docs/scope.md) and [API](../algorithms/affinity/docs/api.md)
for numerical coverage and language adaptations. Dense `KMeans`, `k_means`, and `kmeans_plusplus` are also integrated; see their
[API and numerical limits](../algorithms/kmeans/docs/api.md). `MiniBatchKMeans` is integrated with dense weighted fit and partial_fit; see its
[API](../algorithms/minibatch/docs/api.md). `BisectingKMeans` is integrated with both strategies and tree prediction; see its
[API](../algorithms/bisecting/docs/api.md). Dense `DBSCAN` and `dbscan` are integrated; see the [API](../algorithms/dbscan/docs/api.md).
Dense `MeanShift` and its bandwidth/bin-seed helpers are integrated; see the
[API](../algorithms/mean_shift/docs/api.md). Dense `OPTICS` and its graph,
DBSCAN-extraction and Xi-extraction helpers are integrated; see the
[API](../algorithms/optics/docs/api.md) for metrics, typed sizes/results and
Float64 promotion. Dense `AgglomerativeClustering`, `FeatureAgglomeration`, `linkage_tree`, and
`ward_tree` are integrated; see the [API](../algorithms/agglomerative/docs/api.md).
Dense `Birch` provides CF-tree fitting, streaming updates, prediction, and
subcluster-distance transforms; see its [API](../algorithms/birch/docs/api.md).
Other public estimators remain deferred; the centroid engine is in `shared/centroid_engine`.
All child exports and their owners are listed in [plumbing.md](plumbing.md).

Sparse inputs/outputs, CSR/CSC/COO storage and conversions, sparse precomputed
metrics, connectivity-constrained linkage (including dense connectivity masks
converted to sparse), and nearest-neighbor sparse spectral affinities are deferred
by `session-plan/SPARSITY-SCOPE.md`. Source: sklearn cluster `_kmeans.py`,
`_dbscan.py`, `_agglomerative.py`, `_hdbscan/hdbscan.py`, `_spectral.py` and the
clustering user guide at the pinned reference revision. Dependencies impose the
same sparse exclusion; no local substitute or automatic densification is allowed.
Dense algorithms beyond affinity propagation, KMeans, MiniBatchKMeans, BisectingKMeans,
DBSCAN, MeanShift, OPTICS, Birch, and agglomerative clustering remain future scope. DBSCAN consumes live neighbor radius indices. HDBSCAN nonfinite-row labels remain supported
by the contracts. Sparse absence does not defer dense SVD/eigen requirements.

AffinityPropagation supports its default Euclidean mode. An integer seed is
self-contained; `random_state=None` requires an explicit mutable application RNG.
Agglomerative clustering requires connectivity=None. Future
spectral sessions must select the dense affinity/solver subset approved by
manifold.spectral, excluding nearest_neighbors/precomputed_nearest_neighbors
and sparse solvers; they must document any unsupported default explicitly.
All other dense branches belong to the named child sessions.

Validation commands: `PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run build` and
`PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run test`. Both compile the single
assembled package and two isolated consumers. Test covers Float32/Float64,
rank/feature/empty errors, label noise policies, malformed trees/graphs,
copy isolation, shared views, shared RNG, existing lifecycle/capability tests,
negative import of unimplemented HDBSCAN, absence of DBSCAN.predict and OPTICS.predict, and
missing-source/ownership/export conflicts. Native checks use exact equality on small representable values.
`pixi run parity` builds the family package and compares AffinityPropagation,
KMeans, MiniBatchKMeans, BisectingKMeans, DBSCAN, MeanShift, OPTICS, and agglomerative clustering against the local scikit-learn source in both dtypes, recording
results in `builds/parity.json`, `builds/kmeans-parity.json`,
`builds/minibatch-parity.json`, `builds/bisecting-parity.json`, `builds/dbscan-parity.json`, `builds/mean_shift-parity.json`, `builds/optics-parity.json`, and `builds/agglomerative-parity.json`. DBSCAN compares original sklearn fit/expansion with
independent SciPy radius membership; it does not claim full query-engine parity
or establish parity for deferred estimators. Language
adaptations (typed arrays, optional tree outputs, explicit RNG context and
fluent inherited fit) are documented in plumbing.md and interfaces.md.

The three centroid transformers also expose `get_feature_names_out` as native
string lists. `tests/test_cluster_feature_names.mojo` checks fitted-state and
input-name contracts in both dtypes. The parity command compares exact names
with the local sklearn naming functions and writes
`builds/feature-names-parity.json`.

Dense `Birch` is implemented in `algorithms/birch` and exported by the family
and both isolated consumers. It reuses agglomerative Ward clustering, the
centroid distance kernel, and neighbor queries. See its [API](../algorithms/birch/docs/api.md)
for CF-tree handles, optional labels, incremental behavior, and supported
`n_clusters` configurations. `pixi run parity` writes its comparisons to
`builds/birch-parity.json`.
