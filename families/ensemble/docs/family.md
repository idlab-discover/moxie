# Ensemble family interfaces

Moxie defines ensemble capabilities and implements the dense forest
engine and four public forest estimators. See [forests.md](forests.md) for
usage and [scope.md](scope.md) for the remaining implementation work.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `EnsembleClassifier` | `ClassifierMixin` | Inherited only |
| `EnsembleRegressor` | `RegressorMixin` | Inherited only |
| `ProbabilisticEnsemble` | `EnsembleClassifier, ProbabilityPredictor` | Inherited only |
| `DecisionEnsemble` | `EnsembleClassifier, DecisionFunction` | Inherited only |
| `ForestInspection` | `InputEstimator` | `apply`, `decision_path` |
| `StagedPredictor` | `Predictor` | `staged_predict` |
| `IsolationDetector` | `OutlierMixin, SampleScorer` | Inherited only |

## Behavioral distinctions

Forests, boosting, bagging, voting and stacking have different capabilities. Forest decision_path includes node pointers. Soft-voting probability depends on configuration; staged prediction is not universal. estimators_, feature_importances_ and OOB state follow each algorithm.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../scikit-learn/sklearn/ensemble`.
- Guide and mathematical references: `../../../scikit-learn/doc/modules/ensemble.rst`.
- Shared contracts: `../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`StackingClassifier`, `StackingRegressor`, `BaseForest`, `ForestClassifier`, `ForestRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomTreesEmbedding`, `BaseEnsemble`, `VerboseReporter`, `BaseGradientBoosting`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `BaseBagging`, `BaggingClassifier`, `BaggingRegressor`, `BaseWeightBoosting`, `AdaBoostClassifier`, `AdaBoostRegressor`, `VotingClassifier`, `VotingRegressor`, `IsolationForest`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense extension contracts

[Plumbing](plumbing.md) defines the shared records, engine signatures, ownership
map and dependency assembly. [Scope](scope.md) supersedes legacy sparse demands.
`DenseForestInspection` adds apply without requiring decision_path. All existing
traits remain unchanged. The [forest engine](forest_engine.md) is implemented.
