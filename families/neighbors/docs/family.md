# neighbors family interfaces

This repository contains the dense implementation slices described in
[estimators.md](estimators.md), alongside the original capability scaffold.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `KNeighborsQuery` | `InputEstimator` | `kneighbors` |
| `RadiusNeighborsQuery` | `InputEstimator` | `radius_neighbors` |
| `KNeighborsClassifierAPI` | `ClassifierMixin, ProbabilityPredictor, KNeighborsQuery` | Inherited only |
| `KNeighborsRegressorAPI` | `RegressorMixin, KNeighborsQuery` | Inherited only |
| `RadiusNeighborsClassifierAPI` | `ClassifierMixin, ProbabilityPredictor, RadiusNeighborsQuery` | Inherited only |
| `RadiusNeighborsRegressorAPI` | `RegressorMixin, RadiusNeighborsQuery` | Inherited only |
| `NeighborDensity` | `DensityMixin` | Inherited only |
| `NoveltyDetector` | `OutlierMixin, SampleScorer` | Inherited only |

## Behavioral distinctions

Queries with X=None exclude each training point itself. Radius results are ragged; NeighborResult must represent distances+indices versus indices-only. Sparse graph queries are deferred by the dense scope policy. KDTree and BallTree are implemented at the spatial_trees owner. LOF predicts unseen data only with novelty=True. NearestCentroid is not a neighbor-query estimator.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/neighbors`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/neighbors.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The reference revision and original family commit are recorded in
[PROVENANCE.json](../PROVENANCE.json); license texts are in `../LICENSES`.
Follow citations in the local reference guide and class docstrings when extending
an algorithm. Keep new research under ignored `docs/private/<algorithm>/`.

## Reference classes to investigate

`NearestCentroid`, `KernelDensity`, `NearestNeighbors`, `KNeighborsRegressor`, `RadiusNeighborsRegressor`, `KNeighborsClassifier`, `RadiusNeighborsClassifier`, `NeighborsBase`, `KNeighborsMixin`, `RadiusNeighborsMixin`, `NeighborhoodComponentsAnalysis`, `LocalOutlierFactor`, `KNeighborsTransformer`, `RadiusNeighborsTransformer`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

Concrete dense result records, backend/metric extension traits and dependency
assembly are defined in [plumbing.md](plumbing.md). Read [scope.md](scope.md)
for the authorized sparse exclusions before implementing a child.
