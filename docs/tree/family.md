# tree family interfaces

This repository implements dense numeric decision trees and extra trees alongside
the inherited family contracts. See [scope.md](scope.md) for unfinished branches.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `TreeInspection` | `InputEstimator` | `apply`, `decision_path` |
| `DecisionTreeClassifierAPI` | `ClassifierMixin, LogProbabilityPredictor, TreeInspection` | Inherited only |
| `DecisionTreeRegressorAPI` | `RegressorMixin, TreeInspection` | Inherited only |

## Behavioral distinctions

apply returns leaf indices; decision_path is a sparse node indicator. Multioutput probabilities may be a list of matrices. tree_, feature_importances_, classes_ and n_outputs_ are estimator-specific learned attributes.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/tree`.
- Guide and mathematical references: `../scikit-learn/doc/modules/tree.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`Sentinel`, `DrawTree`, `Tree`, `BaseDecisionTree`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense scaffold extension

Read [plumbing.md](plumbing.md) for the single engine ownership map and exact
native layouts, and [scope.md](scope.md) for sparse exclusions. New dense
implementations use DenseTreeInspection, DenseDecisionTreeClassifierAPI or
DenseDecisionTreeRegressorAPI from `moxie.tree._dense_api`; legacy traits above
remain unchanged. The four public estimators and native export functions come from Moxie's
`src/moxie/tree` package. Its engine modules are also consumed by ensemble and
feature selection through the live Moxie manifest.
