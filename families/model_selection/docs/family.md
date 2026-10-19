# model_selection family interfaces

This repository declares shared family capabilities and assembles concrete
splitter, candidate and CV algorithms from their child owners. See [scope.md](scope.md) for the
implemented subset and remaining queue.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `SearchEstimator` | `BaseEstimator` | Inherited only |
| `PredictiveSearch` | `SearchEstimator, Predictor` | Inherited only |
| `CrossValidator` | `Movable` | `split`, `get_n_splits` |

## Behavioral distinctions

Search estimators expose best_estimator_/cv_results_; prediction requires refit and a suitable wrapped estimator. Splitters are not estimators and do not fit. Parameter search and threshold tuning are separate modules within this family.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/model_selection`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/grid_search.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`ParameterGrid`, `ParameterSampler`, `BaseSearchCV`, `GridSearchCV`, `RandomizedSearchCV`, `BaseSuccessiveHalving`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`, `GroupsConsumerMixin`, `BaseCrossValidator`, `LeaveOneOut`, `LeavePOut`, `KFold`, `GroupKFold`, `StratifiedKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `BaseShuffleSplit`, `ShuffleSplit`, `GroupShuffleSplit`, `StratifiedShuffleSplit`, `PredefinedSplit`, `BaseThresholdClassifier`, `FixedThresholdClassifier`, `TunedThresholdClassifierCV`, `LearningCurveDisplay`, `ValidationCurveDisplay`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Shared plumbing and scope

[plumbing.md](plumbing.md) defines dense layouts, scorer/lifecycle adapters,
results, candidates, refit, child ownership and verified dependency assembly.
[scope.md](scope.md) records tested contracts and authorized sparse deferrals.
Use those contracts before starting any algorithm child.

## Integrated owner

Source and algorithm children are owned by `moxie/families/model_selection`.
Run the focused commands in [README](../README.md) from Moxie.
`IncrementalCVAdapter` opts into learning-curve partial fitting with full-target
class information; ordinary adapters reject incremental mode. LearningCurveDisplay
and ValidationCurveDisplay own copied data and emit native SVG. They require
at least two x values, numeric validation-curve x coordinates and nonempty folds;
Python axes, arbitrary plotting kwargs and metadata routing remain unsupported.
Display plot mutates in place and returns None. Threshold tuning and successive
halving remain explicit raising declarations.
