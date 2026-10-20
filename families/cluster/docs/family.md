# cluster family interfaces

This guide describes the family capability contracts. See [scope.md](scope.md)
for the implemented dense estimators, including DBSCAN and MeanShift, and deferred algorithms.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
fluent mutation returns, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `Clusterer` | `ClusterMixin` | Inherited only |
| `PredictiveClusterer` | `Clusterer, Predictor` | Inherited only |
| `ClusterDistanceTransformer` | `Clusterer, TransformerMixin` | Inherited only |
| `IncrementalClusterer` | `Clusterer` | `partial_fit` |
| `Biclusterer` | `UnsupervisedEstimator` | `get_indices`, `get_shape`, `get_submatrix` |

## Behavioral distinctions

DBSCAN/agglomerative/spectral clustering cannot generally predict unseen samples. KMeans/Birch predict and some algorithms transform or incrementally fit. Biclustering uses row/column membership, not ordinary labels_.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/cluster`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/clustering.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`DBSCAN`, `AgglomerativeClustering`, `FeatureAgglomeration`, `SpectralClustering`, `AgglomerationTransform`, `BisectingKMeans`, `OPTICS`, `Birch`, `MeanShift`, `AffinityPropagation`, `BaseSpectral`, `SpectralCoclustering`, `SpectralBiclustering`, `KMeans`, `MiniBatchKMeans`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Shared implementation boundaries

Read [plumbing.md](plumbing.md) for dense exchange records, shared ownership,
child dependencies and verified assembly. [scope.md](scope.md) records the
accepted dense-only exclusions. The private `_contracts` helpers are implemented;
implemented algorithms and remaining work are listed in [scope.md](scope.md).
