# Dense model-selection scope

The family assembles concrete splitter implementations from `algorithms/splitters`.
All fifteen assigned splitter classes plus `check_cv` and dense `train_test_split`
have native implementations for Int64, Float32/Float64 and String labels and
groups. Non-Int64 labels arrive through `SplitLabels`, which is encoded by the
shared `moxie._labels` owner. See the child's
[scope](../algorithms/splitters/docs/scope.md) for constructor/default contracts,
mutable RNG overloads, eager random cursors, supported ranks/dtypes and the
label differences that the Mojo language forces.

The parent also owns index snapshots/bounds, typed scorer and fit adapters,
clone-fit-score lifecycle, candidate/distribution/refit extension points,
result-column validation and refit guards. These helper contracts preserve the
inherited BaseEstimator and CrossValidator APIs; the validation child builds a serial CV engine on these interfaces. API declarations in `_api.mojo` still raise for unfinished algorithms.
A declared name must not be mistaken for a working estimator.

Sparse inputs/outputs, sparse indexing, sparse prediction assembly and wrapped
sparse estimators remain deferred under session-plan/SPARSITY-SCOPE.md. There is
no implicit densification. Python metadata routing, reflection, heterogeneous
`**kwargs`, warning delivery and plotting objects remain binding work.

## Validation

Builds assemble live Moxie HEAD and declared sibling sources without pins.
Run `pixi run build` and
`pixi run test`, plus the child's build/test/parity tasks. Family tests exercise
both isolated consumer assemblies, source/ownership/export rejection, core
contracts, dense plumbing and the concrete KFold public import. `pixi run
completeness` deliberately fails while public queue entries still map to raising
API declarations. No performance claim is made.

## Remaining queue

ParameterGrid/ParameterSampler are implemented by the search child; see its
[typed candidate scope](../algorithms/search/docs/scope.md). Serial
cross_validate/cross_val_score/cross_val_predict are implemented through explicit
adapters and split lists; see the [validation scope](../algorithms/validation/docs/scope.md)
for tested numeric/String predictions, multioutput responses and class-column
alignment.
`GridSearchCV` and `RandomizedSearchCV` now implement dense serial candidate
evaluation and refitting; see the [search contract](../algorithms/search/docs/search.md).
Serial `validation_curve` shares parameter configuration with search and per-fold
evaluation with CV; it returns train/test scores in parameter-by-split order.
Serial `learning_curve` returns sorted training sizes and train/test
scores, with optional timings, seeded shuffling and explicit weighted adapters.
Serial `permutation_test_score` adds grouped target shuffling, dynamic CV and
empirical p-values. Threshold estimators and successive halving remain deferred. Native displays
produce SVG and typed artist data; they do not implement matplotlib objects. Passing splitter tests
never establishes completion of this family or of the full requested queue.

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

## Unfinished worktrees

The retired checkout preserves ignored `builds/worktrees/thresholds` and
`builds/worktrees/halving`. Thresholds contains an untracked 294-line draft with
no test suite or build command; halving contains manifests and license assets
only. Neither is assembled or claimed as supported. Their draft assets are also
preserved locally in Moxie `builds/model-selection-drafts`; continue that work
in Moxie with API contracts and numerical validation before exporting it.
