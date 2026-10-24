# Clustering in Moxie

Owned by `moxie/families/cluster`. The standalone checkout is retired.

Pure Mojo clustering library derived from moxie `scaffold-v1.1` (`4843da32ff0f`).
The family build exports dense `AffinityPropagation`, `affinity_propagation`,
`KMeans`, `k_means`, `kmeans_plusplus`, `MiniBatchKMeans`, `BisectingKMeans`, `DBSCAN`, `dbscan`, `MeanShift`, `mean_shift`, `estimate_bandwidth`, `get_bin_seeds`, `OPTICS`, `compute_optics_graph`, `cluster_optics_dbscan`, `cluster_optics_xi`, `AgglomerativeClustering`,
`FeatureAgglomeration`, `linkage_tree`, `ward_tree`, and `Birch` from `moxie.cluster`.
See the [KMeans API](algorithms/kmeans/docs/api.md) for weighted initialization,
Lloyd/Elkan fitting, native interfaces and numerical limits. Private centroid
kernels live in `shared/centroid_engine`. The [MiniBatchKMeans API](algorithms/minibatch/docs/api.md)
documents weighted batch fitting and incremental `partial_fit`;
[examples/minibatch.mojo](examples/minibatch.mojo) shows streaming usage.
Dense [DBSCAN](algorithms/dbscan/docs/api.md) supports signed sample weights,
radius-query backends and precomputed distances; see [the example](examples/dbscan.mojo).
Dense [MeanShift](algorithms/mean_shift/docs/api.md) provides bandwidth estimation,
bin seeds, iterative shifting, center merging and prediction; see [the example](examples/mean_shift.mojo).
Dense [OPTICS](algorithms/optics/docs/api.md) provides reachability ordering, Xi hierarchy
and DBSCAN extraction; see [the example](examples/optics.mojo).
Dense agglomerative sample clustering, feature pooling, and tree helpers are
available through `moxie.cluster`. See the [API guide](algorithms/agglomerative/docs/api.md)
and [example](examples/agglomerative.mojo).
Dense [Birch](algorithms/birch/docs/api.md) builds and incrementally updates a CF tree,
with optional Ward clustering of its subcenters; see [the example](examples/birch.mojo).
Other public algorithms remain deferred.

```sh
# From the Moxie repository root:
pixi run cluster-build
pixi run cluster-test
pixi run cluster-parity
pixi run python families/cluster/tools/check.py start
```

`start` clusters a six-point dataset with AffinityPropagation. Family conformance probes live only
in tests and deliberately raise for algorithm operations. `test` also runs the
AffinityPropagation, KMeans, MiniBatchKMeans, BisectingKMeans, DBSCAN, MeanShift, OPTICS, Birch, and agglomerative contracts across two isolated consumers.
`parity` compares the family-built executable with the local scikit-learn reference.

```mojo
from moxie.cluster import AffinityPropagation
import numojo as nm

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(3, 2))
    x[nm.Item(1, 0)] = 1
    x[nm.Item(2, 0)] = 10
    var model = AffinityPropagation(random_state=0)
    var labels = model.fit_predict(x)
    print(labels)
```

See the [AffinityPropagation API](algorithms/affinity/docs/api.md) for dense
Float32/Float64 support, explicit mutable RNGs, copy behavior and native return
records. Sparse inputs and full scikit-learn family parity remain deferred.

- [Family interfaces and reference sources](docs/family.md)
- [Shared estimator contract and Mojo differences](docs/interfaces.md)
- [Dense plumbing, ownership and assembly](docs/plumbing.md)
- [Validated dense scope and sparse exclusions](docs/scope.md)
- [Contribution rules](AGENTS.md)

`SCAFFOLD_BASE` and `SOURCE_PROVENANCE.json` preserve the original revision,
file hashes and scaffold ancestry. Moxie owns current sources and Git history;
the retired checkout retains the original history and ignored local artifacts.
The root dependency manifest assembles the public namespace. Isolated consumer
manifests use current Moxie framework, numerics, RNG and neighbor sources.
Generated packages and numerical reports remain under ignored `builds/`.

The [public API scaffold](docs/api-surface.md) adds scikit-learn declarations with deferred operations.

[BisectingKMeans](algorithms/bisecting/docs/api.md) supports dense weighted
binary splits, both leaf-selection strategies, and tree-based prediction.

Side-by-side notebooks in `notebooks/cluster/` replay the official
`plot_kmeans_digits`, `plot_dbscan`, `plot_mean_shift`, `plot_optics` and
`plot_agglomerative_dendrogram` examples against scikit-learn; rerun them with
`pixi run -e notebooks notebooks notebooks/cluster`. Cluster transformers
(`KMeans`, `MiniBatchKMeans`, `Birch`, `FeatureAgglomeration`) work as
intermediate `moxie.pipeline` steps and inside `make_union`. A clusterer cannot
be the final step of a Moxie `Pipeline`, which requires a supervised predictor.

See [rollup validation](ROLLUP_VALIDATION.json) for the commands, outcomes and
standalone-checkout independence check.
