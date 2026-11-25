# Public imputation API

The family build assembles one `moxie.impute` namespace. Its `SimpleImputer`,
`IntegerSimpleImputer`, `StringSimpleImputer`, `CategoricalNumericImputer`,
`CategoricalStringImputer`, `StringMissingIndicator` and `MissingIndicator`
exports resolve to the concrete implementations in
`algorithms/simple/src/moxie/impute`. Their signatures, supported inputs and
Mojo adaptations are documented in [the algorithm API](../algorithms/simple/docs/api.md).
String imputation uses the family-local `StringMatrix` input and
`StringImputationResult` output types. String indicators return Boolean NDArrays. The family `Imputer` trait remains available.

`KNNImputer` resolves to `algorithms/knn/src/moxie/impute/_knn.mojo`, using
the integrated neighbors query engine. See its [API](../algorithms/knn/docs/api.md)
for supported inputs, callbacks, numerical tolerances and native adaptations.

[`IterativeImputer`](../algorithms/iterative/docs/api.md) implements dense
Float32 and Float64 round-robin imputation and replay with configurable native
regressors. BayesianRidge is the default. Import it directly from
`moxie.impute`; there is no experimental opt-in.

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
| SimpleImputer, typed categorical columns | `CategoricalStringImputer`, `CategoricalNumericImputer` | One category domain per stage, most-frequent and constant; separate indicator stage |
| MissingIndicator, dense numeric data | `MissingIndicator` | Sparse output and unsupported numeric dtypes |
| MissingIndicator, typed categorical columns | `MissingIndicator` | One category domain per call, using the column missing tag |
| MissingIndicator, string data | `StringMissingIndicator` | Heterogeneous categories; sparse output |
| KNNImputer, dense floating data | `KNNImputer` using live neighbors query engine | See the KNN API for callback and dtype adaptations |
| IterativeImputer | Dense deterministic and posterior imputation, native regressors and integer input promotion to Float64 | See the iterative API for uncertainty and RNG contracts |

The family validation includes separate pinned parity runners for simple,
integer, string, KNN and iterative behavior, generic estimator checks, and
two isolated consumer builds.
