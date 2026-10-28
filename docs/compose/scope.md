# Dense composition

Moxie owns the compose implementation under `src/moxie/compose`. All four
names in `sklearn.compose.__all__` have native implementations:

- [ColumnTransformer and its two factories](columns/api.md): selection,
  fitting, weighted concatenation, remainder handling, feature names,
  parameter updates and cloning.
- [TransformedTargetRegressor](target/api.md): target transformation,
  regression, inverse prediction, scoring and the LinearRegression default.
- [ColumnTable and ColumnSchema](named-columns.md): native named numeric
  inputs, name-aligned selection and dtype filtering.

Inputs and outputs use NuMojo Float32/Float64 dense arrays. These are typed
Mojo APIs; the algorithm API pages document constructor and component
adaptations. Export coverage does not establish full scikit-learn parity.
See the [column scope](columns/scope.md) and [target scope](target/scope.md).

Sparse input/output, mixed sparse/dense stacking and density-based sparse
output remain deferred (`sklearn/compose/_column_transformer.py::_hstack`).
Sparse X in TransformedTargetRegressor is unavailable; sklearn also rejects
sparse y. No implicit densification is provided. Use dense components, such
as KBinsDiscretizer with `encode="onehot-dense"`. OneHotEncoder takes typed
categorical columns rather than an `NDArray`, so it cannot be a component
yet; see the [preprocessing composition notes](../../families/preprocessing/docs/api.md#composition). The default
`sparse_threshold=0.3` is valid for wholly dense components; zero explicitly
requests dense output.

Mixed-type native tables, regex selectors, Python dataframe bindings, metadata
routing and HTML display are not implemented. Unsupported regex selection
raises clearly. The typed component contracts reject unavailable capabilities.

## Build and validation

Run from the Moxie repository root:

```sh
pixi run build
pixi run test
pixi run parity
```

The build compiles one generated namespace from `DEPENDENCIES.json`, including
all compose modules, examples and native test drivers. Tests cover public
exports, selections and layouts, both dtypes, parameter ownership, clone/refit,
invalid and unfitted inputs, named tables, parallel components and failed-fit
state preservation. Negative compilation checks reject missing prediction and
block-assembly capabilities. Consumer tests compose real estimators through
ColumnTransformer, TransformedTargetRegressor, Pipeline and DummyRegressor,
and run SimpleImputer, StandardScaler and MinMaxScaler (including as the
remainder) on NaN-containing named and positional columns. The named real-component
contracts compile Float32 and Float64 in separate binaries under the existing
compiler memory cap; both execute the same assertions and parity checks.

Parity scripts under `tests/compose` execute estimator code from the local
scikit-learn revision recorded in [interfaces](../interfaces.md). They cover
column outputs and metadata, named selections, target predictions/scores,
warnings and invalid inputs, and the real consumer compositions. Tolerances
and numerical adaptations are recorded in those scripts. Raw reports stay
under ignored `builds/tests/compose`. Pandas is an oracle-only dependency for
named-column comparisons, never a native runtime dependency.

## Ownership and provenance

Imported from moxie-compose commit
`8574380cafb8308552a7bd1d6f1f04f009a17b8f`. The six implementation modules
are retained unchanged; the package initializer merges their public exports.
`tools/gen_compose_arity.py` maintains the fixed-arity constructor overloads.
No moxie-compose checkout, algorithm-local environment or legacy hash-pinned
assembly launcher is required. Old pin-drift and isolated scaffold assembly
tests are superseded by Moxie's shared assembly tests; estimator contracts and
parity remain part of Moxie's root tasks.

Pipeline and the LinearRegression subset remain external family dependencies.
LinearRegression retains its existing NNLS dependency from the declared
mojo-optimize worktree. Shared dense operations remain in `shared/numerics`;
this integration introduces no numerical operation exception or backend change.
The namespace receipt records consumed bytes and module ownership.
