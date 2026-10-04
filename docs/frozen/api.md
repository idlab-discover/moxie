# FrozenEstimator

`from moxie.frozen import FrozenEstimator, FrozenParams, is_clusterer` provides
a pure Mojo wrapper around any `BaseEstimator & Deinitable`. `FrozenEstimatorAPI` remains
the family contract. Shared estimator contracts are unchanged.

```mojo
var model = DummyRegressor()  # mean-only implementation in examples/frozen/
model.fit(X, y)
var frozen = FrozenEstimator(model^)
frozen.fit(X, different_y)   # checks fitted state, never calls model.fit
var same_state = clone(frozen)
var predictions = same_state.predict(X)
print(frozen.estimator().constant_.value())
```

After `pixi run build`, run the complete example with
`builds/examples/frozen/frozen`.

## Ownership and fitted state

Construction consumes an estimator (`model^`). To retain an independent source
value, explicitly pass `model.copy()` when its copy semantics provide independent
state. The wrapper does not clone the incoming estimator: normal estimator
cloning would discard the fitted buffers.

The wrapper owns an `ArcPointer` internally. Its `estimator()` accessor returns
an immutable borrow. Wrapper copies, `clone(frozen)`, and `get_params()` records
share the same allocation and preserve fitted state, without calling the inner
clone implementation or copying its learned buffers. They remain valid after
the original wrapper goes out of scope. The last owner releases the allocation.
Underscore-prefixed storage is internal; public accessors do not permit mutation.
Explicit `partial_fit` calls do mutate the shared estimator, as described below.
As with other immutable borrows, this does not make an estimator's own internally
shared mutable buffers deeply immutable.

`set_params(estimator=new_model^)` replaces the estimator in that wrapper only.
Existing clones and saved parameter records keep the previous allocation.
Replacement must have the same concrete estimator type; construct a new wrapper
to change types. Both construction and replacement accept unfitted estimators;
subsequent `fit` then raises `NotFittedError`. Fitted-state queries delegate to
the current estimator, including for stateless estimators.

`fit` checks **only fitted state**, exactly as the reference does. It ignores
features and targets, including empty, inconsistent, or otherwise invalid data.
The supervised, unsupervised, and weighted-fit trait signatures are supported
when present on the wrapped estimator. A generic two-argument overload also
accepts ignored values outside its input and target types. No fit overload calls
the wrapped estimator or changes learned state.

## Parameters and capability forwarding

`get_params(deep=True)` and `get_params(False)` return the same typed
`FrozenParams[E]` shape, containing only the estimator ownership handle. Inspect
the estimator through `params.estimator()`. Restore it with
`frozen.set_params(params)` or construct another wrapper with
`FrozenEstimator[E](params)`. `set_params()` without arguments is a no-op.

Inner parameter records and `estimator__parameter` keyword updates are rejected
at compile time. Calling mutating methods through either read-only accessor is
also rejected. This is deliberately stricter than Python's mutable
`frozen.estimator` attribute.

| Wrapped capability | Forwarded methods and behavior |
| --- | --- |
| `Predictor` | `predict`, preserving the associated prediction type |
| `SupervisedScorer` | `score`, including optional sample weights |
| `ClassifierMixin`, `RegressorMixin` | Corresponding trait identity and scoring |
| `ProbabilityPredictor`, `LogProbabilityPredictor` | `predict_proba`, `predict_log_proba` |
| `DecisionFunction` | `decision_function` |
| `TransformerMixin`, `InverseTransformer` | `transform`, `inverse_transform` |
| `SampleScorer`, `DensityMixin` | `score_samples`, estimator-specific density `score` |
| `OutlierMixin` | Corresponding trait identity and inference methods |

Each capability is conditional on the wrapped type's trait conformance. Missing
capabilities fail compilation, and configuration-dependent errors from supported
methods propagate unchanged. Associated result types, shapes, and dtypes are
preserved. There are no conversions or numerical kernels in the wrapper.

`fit_predict` and `fit_transform` are not exposed.

### Incremental fitting

Like the pinned Python implementation, freezing disables `fit` but does **not**
disable `partial_fit`. Wrapping an `IncrementalRegressor`, `IncrementalClassifier`,
or `IncrementalTransformer` conditionally preserves that trait and forwards its
`partial_fit` signature. Classifier `classes` defaults to `None`; transformer
`y` defaults to `None`. The wrapped method performs all validation, can initialize
an unfitted estimator, and propagates errors unchanged. There is no wrapper
fitted-state precheck, no no-op substitute, and no rollback of inner mutations.

Updates are visible through existing copies, clones, saved parameter records,
and nested wrappers sharing the same estimator allocation. `fit` remains a no-op
after an update. Replacing an estimator still changes only that wrapper's handle.
Accessors return immutable borrows, but the allocation is no longer assumed to
remain unchanged across calls to `partial_fit`. Do not retain interior views
across updates through another shared handle or perform concurrent reads and
updates; reference counting does not synchronize estimator operations.

These overloads return a mutable borrow of this wrapper, following the shared
fluent contract. Arbitrary
extra arguments and incremental sample weights are not part of the shared
incremental traits and are not synthesized. Nonincremental estimators reject
`partial_fit` at compile time.

### Cluster identity

Use `from moxie.frozen import is_clusterer` and `is_clusterer(model)` for native
cluster identity. The query recognizes `ClusterMixin` on ordinary estimators
and unwraps nested `FrozenEstimator` values, including unfitted ones. It returns
false for nonclusterers. It neither fits the estimator nor exposes `fit_predict`.

This family-local query adapts sklearn's `sklearn.base.is_clusterer` location:
the shared Mojo `ClusterMixin` bundles identity with required `fit_predict`.
Frozen wrappers therefore still do not conform to that trait. Shared tags and
a framework-wide identity-only trait remain outside this family change. A
predictive clusterer's independent `Predictor` capability can still be forwarded.

## Native differences and scope

- Python clone returns the same wrapper object. Mojo clone returns a value
  sharing its estimator allocation; replacing one wrapper's estimator leaves
  other wrapper values unchanged.
- Learned attributes and custom read-only methods use `frozen.estimator()`;
  arbitrary attribute forwarding, indexing, and inference overloads outside
  the shared traits are not synthesized.
- Fit, partial fit, and parameter changes return a mutable borrow of this wrapper.
  Nested parameter rejection occurs at compile time rather than as a Python
  `ValueError`.
- Arbitrary fit `*args`/`**kwargs`, dynamic metadata routing, Python tags
  (including `_skip_test`), bindings, and serialization remain outside the
  native interface. Classification and regression identity use Mojo traits.

The example's real `DummyRegressor` implements only unweighted mean fitting and
R2 scoring for single-output finite Float64 targets. Features are a
`List[Float64]` representing one feature; values are ignored, as in the reference.
Its learned `constant_` is an optional scalar, not a `(1, 1)` array. It supports
empty prediction inputs; alternative strategies and score weights raise clear
errors. It is example support, not an implementation of the whole dummy family
or a new export from `moxie.dummy`.

## Verification and reference

The reference is local scikit-learn `1.10.dev0`, revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, especially
`sklearn/frozen/_frozen.py`, `sklearn/frozen/tests/test_frozen.py`, and
`sklearn/dummy.py`. The source is BSD-3-Clause; this wrapper implements its
behavior using native ownership and traits.

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
builds/examples/frozen/frozen
```

The native suite checks shared contracts, unfitted errors, repeated fitting,
clone lifetime and shared allocation, parameter replacement and restoration,
nested wrappers, conditional traits, successful forwarding, and error propagation.
Negative compilation checks cover unavailable methods (including `partial_fit`
on nonincremental estimators), nested parameters, and mutation through the public
accessors. Focused stateful probes cover all three incremental signatures,
shared updates, nested wrappers, delegated errors, and cluster identity.
Synthetic capability probes test dispatch only; the mean-regressor example supplies real learned predictions.

For reproducible reference comparisons, place the pinned scikit-learn checkout
at `../scikit-learn` (or pass `--sklearn-root PATH`), then run:

```sh
pixi run parity
```

The runner loads the **pinned local FrozenEstimator and DummyRegressor source
files**; installed sklearn supplies their supporting infrastructure. It reports
the support package versions and verifies the local Git revision. Forty
learned-model observations cover prediction, empty prediction, repeated fit,
cloning, R2, and estimator replacement/restoration across five target
distributions. Eleven additional synthetic observations compare probability,
decision, transformation and density dispatch. Prediction
fixtures are exactly representable and require exact equality; R2 allows
`rtol=atol=1e-12` for Float64 reduction-order differences. Python and its numerical
dependencies execute only this comparison tooling. This evidence establishes the
documented native scope, not full Python API parity.

The focused incremental/identity comparison uses synthetic stateful probes, not
additional estimator integrations. It loads both `_frozen.py` and `base.py` from
the same pinned checkout and compares ten integer observations exactly. Run it
from this project with a reference Python environment:

```sh
pixi run python tests/frozen/compare_frozen_updates.py --sklearn-root ../scikit-learn
```

Moxie's `pixi run parity` includes both comparisons automatically.

## Upstream test coverage

| Reference contract | Native evidence |
| --- | --- |
| `test_frozen_methods` | `test_frozen_capabilities.mojo` checks trait identity, inference dispatch and delegated errors; `compare_reference.py` compares learned predictions and scores |
| `test_composite_fit` | Nine negative compilation checks include rejection of `fit_predict` and `fit_transform` |
| `test_clone_frozen`, `test_check_is_fitted` | `test_frozen.mojo` covers shared allocation, clone lifetime, unfitted errors and repeated no-op fit |
| `test_frozen_params` | Typed parameter records, restoration, same-type replacement, and compile-time rejection of nested updates |
| `test_frozen_tags` | Classifier, regressor and outlier trait identity plus recursive cluster identity; arbitrary Python tags are unsupported |
| `test_frozen_metadata_routing` | Unsupported: dynamic metadata routing and indexing require a separate native design |

Capability probes establish forwarding contracts, not numerical coverage of every
wrapped estimator. Python object identity, arbitrary attribute/index forwarding,
cross-type replacement and dynamic arguments retain the differences listed above.
The integrated implementation comes from `moxie-frozen` commit
`ee63d6cd8093e3d1611bd0d3427359ad1fd58b98`; its original sibling checkout is not
a build or test dependency.
