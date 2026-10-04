# covariance family interfaces

This repository implements the dense covariance family and its shared capabilities.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `CovarianceEstimator` | `UnsupervisedEstimator` | `get_precision`, `mahalanobis`, `score` |
| `RobustCovarianceDetector` | `OutlierMixin, SampleScorer` | Inherited only |

## Behavioral distinctions

Covariance estimators expose covariance_, location_, optional precision_ and Gaussian
log-likelihood scores. EllipticEnvelope reuses robust covariance state but its score
is label accuracy; its score_samples is negative squared Mahalanobis distance.
Not every covariance estimator predicts. See [plumbing](plumbing.md) for shared
array ownership, complete child ownership and engine extension points, and
[dense scope](scope.md) for the authorized exclusions.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/covariance`.
- Guide and mathematical references: `../scikit-learn/doc/modules/covariance.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`EmpiricalCovariance`, `EllipticEnvelope`, `MinCovDet`, `BaseGraphicalLasso`, `GraphicalLasso`, `GraphicalLassoCV`, `ShrunkCovariance`, `LedoitWolf`, `OAS`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.
