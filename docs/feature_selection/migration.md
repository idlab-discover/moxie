# Integration requirements

Moxie owns the implementation, tests, examples and documentation. All 19
reference exports and native extension types remain under `moxie.feature_selection`.
Export coverage does not establish full API parity.

## Fluent estimator contracts

The family follows Moxie's mutable-borrow return contract for `fit`, `set_params`
and supported `partial_fit` methods. All six univariate selectors and
VarianceThreshold support both concrete and generic chaining. Model selectors
and their adapters use the same signatures. Native `fit_named` and observer
helpers remain in-place methods without a fluent return.

```mojo
var selected = selector.set_params(params).fit(X, y).transform(X)
```

The returned value borrows the original estimator. It does not clone learned
state. A failed chain retains the effects of calls that already succeeded;
individual fits keep their existing failure-state behavior.

## Native helper functions

For supervised selector helpers, use the shared estimator trait:

```mojo
from moxie.base import SupervisedEstimator

@no_inline
def fit_selector[T: SupervisedEstimator](
    mut selector: T, X: T.Input, y: T.Target
) raises:
    _ = selector.fit(X, y)
```

Mojo 1.0 rejects the fluent `fit` call in some helper parameters spelled as a
concrete `RFECV` alias, reporting different receiver types with identical printed
names. This also occurs in the pre-integration family namespace. The classifier
parity generator uses the generic helper above and retains its real estimators,
fixtures and comparisons. SelectFromModel uses its separate optional-target
contract. Broader concrete-wrapper compatibility remains limited.

RFECV parity fixtures put each phase's CV-result assertions in a separate
`@no_inline` function. Keeping both phases in `main` made Mojo 1.0 emit an
impossible stack-frame size and a crashing executable for Float32 features
with three Int64 classes. The split retains both fits, all result columns,
exact assertions and numerical comparisons under the same compiler limits.

## External acceptance cases

PCA and IncrementalPCA must implement the live `BaseEstimator.set_params` and
`UnsupervisedEstimator.fit` signatures. IncrementalPCA must also implement
`IncrementalTransformer.partial_fit`. Each returns `ref[self] Self` with an
explicit mutable receiver origin and immutable origins for required inputs.
Optional targets keep ordinary Optional arguments. Alternate configuration and
`check_input` overloads must retain their documented behavior.

Acceptance requires the actual decomposition implementation, not a replacement:
`IncrementalPCAImportance` must compile, copy fitted state independently, and
support `SelectFromModel.partial_fit` across multiple batches. Run
`tests/feature_selection/test_from_model_incremental.mojo`,
`tests/feature_selection/parity_from_model_incremental.py`, and both isolated consumers through the
root commands after the external migration.

The live nearest-neighbor implementation imports its base contracts and spatial
tree modules. Every family and consumer manifest must include those modules
from their actual owner. The tree importance adapter copies the live tree's
encoder along with its fitted tree data; it must preserve independent fitted
snapshots and native string/Int64 class labels.

## Validation commands

Run `pixi run build`, `pixi run test`, and `pixi run parity` for full integration.
Test includes both isolated consumer namespaces and real model fits. Do not
replace a failed integration result with subset evidence.

Use `pixi run feature-selection-build`, `pixi run feature-selection-test` and
`pixi run feature-selection-parity` from Moxie for full family validation without
running unrelated families. These include the real decomposition adapter and
both isolated consumers. Focused diagnostic checks are not substitutes.

Sparse support, Python metadata routing, container output, warning transport,
and dynamic reflection remain deferred as described in [scope](scope.md).
Broader numerical and estimator-wrapper parity remains separate from fluent
return compatibility.
