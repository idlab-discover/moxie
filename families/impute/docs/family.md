# impute family interfaces

This integrated family implements the dense contracts in [the API guide](api-surface.md).
IterativeImputer remains deferred.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `Imputer` | `UnsupervisedEstimator, TransformerMixin, FitTransformer` | Inherited only |

## Behavioral distinctions

Missing-value conventions, retained empty features, indicator columns and feature-name handling are observable behavior. All four estimators, including IterativeImputer, are directly public in the pinned reference checkout.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/impute`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/impute.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`KNNImputer`, `SimpleImputer`, `MissingIndicator`, `IterativeImputer`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

Shared ownership, exact native layouts and dependency assembly are specified in
[plumbing.md](plumbing.md). See [scope.md](scope.md) for dense-only exclusions.
