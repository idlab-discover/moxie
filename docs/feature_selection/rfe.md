# Recursive feature elimination

`RFE` repeatedly clones and fits an estimator on the remaining columns, removes
the least important columns, and fits a fresh final estimator on the selected
subset. It uses the same dense arrays, model adapters and importance aggregation
as [SelectFromModel](from_model.md). The internal path observer supplies the
elimination states used by [RFECV](rfecv.md). RFE
does not introduce a CV engine.

```mojo
from moxie.feature_selection import RFE, FeatureCount, LassoImportance
from moxie.linear_model import Lasso

var selector = RFE(
    LassoImportance(Lasso(alpha=0.1)),
    n_features_to_select=FeatureCount(2),
    step=FeatureCount(1),
)
selector.fit(X, y)
var selected = selector.transform(X)
var predictions = selector.predict(X)
var ranking = selector.ranking_()
```

See [the runnable example](../../examples/feature_selection/rfe.mojo). Both Float32 and Float64 are
supported; the adapter and feature dtypes must agree. The target type belongs
to the estimator. Supplied adapters wrap real Lasso and ElasticNet estimators
(including dense multi-output and sample weights) and single-output
DecisionTreeClassifier with independent numeric or string labels. Estimator
restrictions still apply.

## Parameters and fitted state

Constructor defaults match sklearn: `n_features_to_select=None`, `step=1`,
`verbose=0`, `importance_getter="auto"`. Native `FeatureCount` is an explicit
`Variant[Int, Float64]`: integer counts and floating-point fractions remain
distinct. None selects half the original features (rounded down). Fractions
for selection must lie in (0,1]; step fractions must lie in (0,1). Fractional
steps are rounded down using the **original** input width, with a minimum of
one feature per iteration. Integer counts above the width retain every feature.
An accepted selection fraction that rounds down to zero reaches a zero-column
final fit and is rejected by the supplied estimators, as in sklearn.

Fit requires at least two input features. Each iteration aggregates squared
importances across output rows. Equal importances are eliminated in original
column order. Selected features have rank 1; earlier eliminations receive higher
ranks. All fits use fresh estimator clones, including the final refit, even if
the configured estimator was fitted or has warm starts enabled.

Learned names use owned accessors: `estimator_()`, `support_()`, `ranking_()`,
`n_features_()`, `n_features_in_()`, `feature_names_in_()`. Modifying a returned
array, list or estimator does not mutate the selector. `fit_named(X, y, names)`
records names; ordinary fit replaces them with unnamed features. Fitting commits
all learned state together after success; a failed fit preserves previous state.
`set_params` changes configuration without refitting. `get_params` returns a
`RFEParams` record holding an independent estimator configuration/state copy;
edit its typed nested estimator parameters and apply `set_params(record)`.
`__sklearn_clone__` recursively clones configuration into an unfitted selector.

`fit(X, y, sample_weight=weights)` forwards weights to every fit.
`fit_transform(X, y.copy())` fits, then delegates transformation to the public mixin, and requires a target.
`predict` and `score` select the input columns and delegate to the fitted adapter;
score retains the wrapped estimator's semantics and optional score weights.
Optional probability and decision methods are conditionally exposed through the
corresponding Moxie capability traits. Inference adapters must declare
`Input = NDArray[dtype]`; incompatible inputs are rejected at compilation. No probability methods are exposed for
Lasso or ElasticNet. `classes_()` forwards owned labels for adapters implementing
`SelectionClassLabels`, preserving their order. DecisionTreeClassifierImportance
supplies production class labels, probabilities and log probabilities. A
test-only routing probe additionally covers decision-function forwarding.

## Internal elimination-path seam

`RFE._fit_with_observer(X, y, observer, names=[], sample_weight=None)` runs the
same elimination as ordinary fit. Its mutable observer implements
`_RFEStepObserver` from `moxie.feature_selection._rfe`, with an associated
`Estimator` exactly equal to the selector's adapter type and one method:

```mojo
def observe(mut self, estimator: Self.Estimator,
            selection: SelectionState, ranking: List[Int]) raises:
    ...
```

Every call borrows a fresh fitted estimator and the current selection in original
column coordinates. Calls occur before importance extraction and pruning, then
once after the final successful fit. Feature counts descend; the final target
is always visited, including when the last step removes fewer columns than the
configured step. If no columns are removed, only the final fit is observed.
`selection.selected_count()` gives the count; `selection.get_support()` gives an
owned mask or index list; `selection.transform(X_test)` selects held-out columns
in the same order used for fitting. Rankings cover all original columns.

An observer can apply a typed scorer and retain owned score/mask/ranking
snapshots. Selection accessors copy their outputs; copy `ranking` before
retaining it. The fitted estimator is borrowed: retaining its state requires an
explicit independent copy. The engine does not impose prediction or scoring
capabilities on ordinary RFE adapters, and adds no fits to collect observations.
Ordinary `fit` and `fit_named` use a no-op observer and allocate no path history.
This is an internal family extension, not a new sklearn constructor parameter.

Observer and fitting errors propagate. RFE commits learned state only after all
fits and observations succeed, preserving previous fitted state on failure. The
observer is caller-owned: it is not reset or rolled back and may hold a partial
path after an error. Use a fresh observer per fold and publish its results only
after the whole fold succeeds. RFECV uses this seam for fold orchestration,
score aggregation, feature-count selection and final refit.

The runnable [held-out path example](../../examples/feature_selection/rfe_path.mojo) implements an
observer using real Lasso fits and independent snapshots. Run `pixi run test`
to execute it. Custom observers can use other scorers, metadata and estimator
capabilities without changing the elimination loop.

## Native differences and scope

The public [SelectorMixin](selector-mixin.md) supplies mapping methods. Sparse
input/importances are deferred; no implicit densification occurs. Input matrices
are borrowed, and selected matrices are new owned arrays, including strided
inputs. Numeric coercion, object/integer matrices, Python warning transport,
metadata routing, arbitrary fit/predict keyword dictionaries, pandas output and
Python callable/attribute-path reflection remain unsupported. Counts above the
input width retain all features without emitting a Python warning. Custom
importance getters belong in typed adapters' `importances(getter)` methods.
Learned accessors and in-place fitting follow [the native interfaces](../interfaces.md).

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity`. Both isolated
consumer assemblies exercise the public RFE and SelectorMixin exports.
`tests/feature_selection/test_rfe.mojo` covers lifecycle, parameters, invalid inputs, names, owned
outputs, strides, stable ties, weighted fits, multi-output fitting, prediction,
scoring and unfitted cloning with real models in both dtypes.

`tests/feature_selection/parity_rfe.py` executes the local sklearn `RFE._fit` body at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The validation-only Python estimator
uses the exact orthogonal-design Lasso/ElasticNet solution; native fixtures use
the real Mojo estimators. This isolates elimination parity without substituting
a native test optimizer. Masks, rankings and transformations compare exactly;
coefficient/prediction tolerances are Float32 rtol=3e-5, atol=2e-7 and Float64
rtol=3e-8, atol=2e-12, allowing bounded optimizer and arithmetic rounding.
The harness does not execute sklearn's full validation pipeline or establish
parity for every estimator or ill-conditioned design. No performance claim is
made. Reference mathematics: Guyon et al., *Gene Selection for Cancer
Classification using Support Vector Machines*, Machine Learning 46 (2002),
389–422, DOI 10.1023/A:1012487302797.

`tests/feature_selection/test_rfe_path.mojo` checks observer timing, original-coordinate snapshots,
weighted held-out scoring, fractional/clipped steps, no-elimination paths,
repeated calls, callback/fit failures, and equivalence with ordinary fit.
A negative compilation test rejects an observer for a different estimator type.
`tests/feature_selection/parity_rfe_path.py` additionally executes the pinned `RFE._fit(step_score)`
body and compares every observed count, support and ranking exactly, and weighted
held-out R2 with the dtype tolerances above. It includes Lasso/ElasticNet,
multiple outputs, weights, tied importances and reversed columns. The same
analytic-oracle limitations apply; this is not an RFECV implementation or test.

`DecisionTreeClassifierImportance` adds a production classifier adapter for
dense Float32/Float64 features and native string/Int64 targets. See
[classifier selection](classifier-selection.md) for the supported single-output
contract, dependency provenance and parity checks.
