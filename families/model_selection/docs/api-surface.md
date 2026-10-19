Concrete splitter, candidate, grid/random-search, validation/learning-curve, permutation-test and CV exports now resolve to the child-owned `_split`, `_search_parameters`, `_search` and `_validation` modules through
`EXPORTS.json` and shared assembly. Its tested native interfaces supersede the
corresponding retained declarations below. Other `_api` names still raise;
run `pixi run completeness` to list them. See [scope.md](scope.md).

# Public API and retained declarations

Public names, constructor defaults and inherited callable signatures follow the
local `scikit-learn/sklearn/model_selection` source. Newly declared constructors and
operations raise Mojo `Error` with a `NotImplementedError` prefix. Existing
family traits and implementations are preserved; no numerical operation is added.

Native heterogeneous inputs use `Some[Movable]`. Configuration and learned
attributes and deferred results use the family-local opaque `APIValue` marker;
this marker supplies no array, estimator or Python runtime implementation.
Python `**kwargs` is an optional named native argument, retaining its source name.
Fit and parameter updates mutate in place without fluent returns. Properties are
opaque fields. Inherited public methods are declared directly on each struct.
Dynamic metadata request generation and Python reflection remain unsupported.
Exact Python signatures accompany the declarations.

Public exports: `BaseCrossValidator`, `BaseShuffleSplit`, `FixedThresholdClassifier`, `GridSearchCV`, `GroupKFold`, `GroupShuffleSplit`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`, `KFold`, `LearningCurveDisplay`, `LeaveOneGroupOut`, `LeaveOneOut`, `LeavePGroupsOut`, `LeavePOut`, `ParameterGrid`, `ParameterSampler`, `PredefinedSplit`, `RandomizedSearchCV`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `ShuffleSplit`, `StratifiedGroupKFold`, `StratifiedKFold`, `StratifiedShuffleSplit`, `TimeSeriesSplit`, `TunedThresholdClassifierCV`, `ValidationCurveDisplay`, `check_cv`, `cross_val_predict`, `cross_val_score`, `cross_validate`, `learning_curve`, `permutation_test_score`, `train_test_split`, `validation_curve`.

Compilation uses a local `builds/api-surface/imports` tree with the existing
assembled dependencies overlaid by `src`, so the current API is compiled without
rewriting assembled dependency output:

```sh
pixi run mojo precompile builds/api-surface/imports/moxie \
  -I builds/api-surface/imports -o builds/api-surface/moxie.mojoc
```

No tests were added or run for these API declarations.

NaN defaults use the native IEEE Float64 NaN value. The implemented learning
curve uses the five reference fractions when its native `train_sizes` argument
is omitted; see the [typed validation API](../algorithms/validation/docs/scope.md).
Experimental halving-search names are declared directly; there is no Python
experimental import-enablement mechanism.

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
