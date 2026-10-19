# multiclass family interfaces

Implemented: [`OneVsRestClassifier`](one_vs_rest.md),
[`OneVsOneClassifier`](one_vs_one.md) and
[`OutputCodeClassifier`](output_code.md) for dense numeric or string
single-label targets. OvR also supports dense multilabel indicator targets.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `PairwiseInput` | Family-local capability | `pairwise_input(self) -> Bool`, current configuration |
| `MulticlassClassifier` | `ClassifierMixin` | Inherited only |
| `MulticlassDecision` | `MulticlassClassifier, DecisionFunction` | Inherited only |
| `MulticlassProbability` | `MulticlassClassifier, ProbabilityPredictor` | Inherited only |

## Behavioral distinctions

One-vs-rest, one-vs-one and output-code wrappers clone component estimators. Available methods depend on wrapped estimator capabilities; probability and partial_fit are not universal.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/multiclass.py`.
- Guide and mathematical references: `../scikit-learn/doc/modules/multiclass.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/multiclass/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.
