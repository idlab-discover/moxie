# Public imputation API

The family build assembles one `moxie.impute` namespace. Its `SimpleImputer`,
`IntegerSimpleImputer`, `StringSimpleImputer`, `StringMissingIndicator`, and `MissingIndicator` exports resolve to the concrete implementations in
`algorithms/simple/src/moxie/impute`. Their signatures, supported inputs and
Mojo adaptations are documented in [the algorithm API](../algorithms/simple/docs/api.md).
String imputation uses the family-local `StringMatrix` input and
`StringImputationResult` output types. String indicators return Boolean NDArrays. The family `Imputer` trait remains available.

`KNNImputer` resolves to `algorithms/knn/src/moxie/impute/_knn.mojo`, using
the integrated neighbors query engine. See its [API](../algorithms/knn/docs/api.md)
for supported inputs, callbacks, numerical tolerances and native adaptations.

`IterativeImputer` remains a declaration in `_api.mojo`.
Its constructor and operations raise `NotImplementedError` through Mojo
`Error`. Its name and parameter defaults mirror the pinned scikit-learn
source, but it provides no numerical behavior. Opaque declaration fields use
`APIValue`; these are not fitted model implementations.

`EXPORTS.json` defines the public exports. The family and algorithm dependency
manifests declare the same concrete modules, with live family-owned source and
current dependency owners. No source overlay or separate API-only compile
step is needed. `pixi run build` compiles all public modules, examples and test
drivers; `pixi run test` exercises public imports, estimator contracts and two
isolated consumer namespaces. `pixi run parity` compares the family namespace
against the reference estimator bodies.

Python reflection, metadata routing, serialization and pandas/Polars output
integration are outside the native API. The documented in-place `fit` and typed
parameter records follow the [shared interface contract](interfaces.md).

## Coverage and migration boundaries

| Reference capability | Native implementation | Remaining limits |
| --- | --- | --- |
| SimpleImputer, dense floating data | `SimpleImputer`, including native callback strategies | Documented ownership and inverse edge adaptations |
| SimpleImputer, dense integer data | `IntegerSimpleImputer` | Typed output and range adaptations |
| SimpleImputer, string data | `StringSimpleImputer` | Retained empty most-frequent columns requiring numeric zero; inverse; heterogeneous categories |
| MissingIndicator, dense numeric data | `MissingIndicator` | Sparse output and unsupported numeric dtypes |
| MissingIndicator, string data | `StringMissingIndicator` | Heterogeneous categories; sparse output |
| KNNImputer, dense floating data | `KNNImputer` using live neighbors query engine | See the KNN API for callback and dtype adaptations |
| IterativeImputer | Deferred declaration only | Entire iterative algorithm and real-regressor integration |

The integrated family retains these limits and preserves the simple, integer, string, and KNN
parity runners, generic estimator checks, and both isolated consumer builds.
Compilation of the IterativeImputer declaration is not algorithm evidence.

IterativeImputer's default estimator requires a concrete native
`moxie.linear_model.BayesianRidge[dtype]` from the linear-model owner.
Moxie now owns a concrete dense BayesianRidge implementation.
IterativeImputer has not yet been implemented against it. Required operations are
configuration-only cloning, `fit(X, y)` for dense `(n, p)` predictors and `(n,)`
targets, `predict(X)` returning `(n,)`, and a typed equivalent of
`predict(X, return_std=True)` returning mean and standard-deviation vectors.
Each fitted model must retain independent learned state for sequence replay.
Acceptance must compare both prediction modes against the pinned sklearn
BayesianRidge on constant, rank-deficient, and ordinary dense designs, then run
an IterativeImputer round with that real model and replay it on new rows without
refitting. Posterior sampling also requires the live shared RandomState owner.
These requirements do not block the concrete simple, indicator, or KNN exports.
