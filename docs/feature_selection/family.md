# feature_selection family interfaces

This repository implements dense selectors and scoring functions alongside the
family capability traits. See [public API coverage](api-surface.md) for the
implemented surface and remaining parity limits.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `FeatureSelector` | `TransformerMixin, InverseTransformer` | `get_support` |
| `SupervisedSelector` | `SupervisedEstimator, FeatureSelector` | Inherited only |
| `UnsupervisedSelector` | `UnsupervisedEstimator, FeatureSelector` | Inherited only |

## Behavioral distinctions

Support must represent a Boolean mask or index vector. Supervised selection requires targets; VarianceThreshold does not. Meta-selectors clone and route configuration to their estimator.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/feature_selection`.
- Guide and mathematical references: `../scikit-learn/doc/modules/feature_selection.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`VarianceThreshold`, `SelectPercentile`, `SelectKBest`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `GenericUnivariateSelect`, `SelectorMixin`, `SequentialFeatureSelector`, `RFE`, `RFECV`, `SelectFromModel`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense shared plumbing

Before implementing children, read [plumbing.md](plumbing.md) for the exact
support state, score/policy/importance extension points, ownership map and
verified dependency assembly. [scope.md](scope.md) records dense-only limits
and sparse exclusions. The helpers provide shared mechanics; all selectors
consume the same family-owned mapping implementation.
