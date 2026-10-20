# Dense FeatureUnion

`moxie.pipeline.FeatureUnion` fits transformers on the same dense input and
concatenates their outputs in declaration order. Execution is pure Mojo over
NuMojo `NDArray[Float32/Float64]`. It implements the foundation
`UnsupervisedEstimator`, `FitTransformer`, and `TransformerMixin` traits, so it
can be used directly as a transformer in this family's `Pipeline` or another
FeatureUnion.

## Construct a union

Use `union_step` for existing estimators, `UnionSentinel` for passthrough/drop,
and `UnionPair` to combine heterogeneous branches:

```mojo
from moxie.pipeline import FeatureUnion, UnionPair, UnionSentinel, union_step
from moxie._pipeline import StepMode

# left and right are concrete transformers from their owning model families.
var branches = UnionPair(
    union_step[DType.float64]("left", left^),
    UnionPair(union_step[DType.float64]("right", right^),
              UnionSentinel("unused", StepMode.drop)),
)
var weights: Dict[String, Float64] = {"left": 0.5, "right": -2.0}
var combined = FeatureUnion(branches^, transformer_weights=weights^)
var training_features = combined.fit_transform(X, y.copy())
var test_features = combined.transform(X_test)
```

`union_step` and `UnionSentinel` default to Float64; select Float32 explicitly
for Float32 estimators. Every branch must have the same dtype and use
`NDArray[dtype]` for input, output, and any target. A single branch needs no pair.
Nest pairs for arbitrary arity: these pairs form one flattened branch list, so
names must be unique across the entire tree. Wrapping an actual FeatureUnion
inside `union_step` instead creates a nested estimator with its own names.

The constructor preserves the reference options/defaults:
`n_jobs=None`, `transformer_weights=None`, `verbose=False`, and
`verbose_feature_names_out=True`. This implementation executes sequentially:
`n_jobs=None` and `1` are supported; other values raise explicitly. Verbose mode
prints branch progress, using a native log format. The statically typed branch
tree replaces Python's heterogeneous list of `(name, estimator)` tuples.

See [the runnable example](../../examples/feature_union.mojo), which concatenates
original and half-weighted features and uses the resulting union in a Pipeline
with the implemented DummyRegressor. Run:

```sh
pixi run build
./builds/feature_union_example
```

## Fitting, ownership and validation

Branches accept existing foundation traits; no change to their implementation
or `CompositionStep` conformance is required. They must implement transform and
either fit or fit_transform. `fit_transform` prefers `FitTransformer`; otherwise
it calls fit followed by transform. `fit` calls fit alone and does not perform a
transformation. A branch supporting only fit_transform works with fit_transform
but raises on fit, as in the reference. Supervised branches require a supplied
target; optional-target branches also accept `y=None`.

Every branch borrows the same input and target. There is no clone-on-fit.
Fitted state stays in the constituent estimators. Repeat and failed fits obey
those estimators' state rules; exceptions do not roll back earlier branches.
Output shape checks run after branch execution. Each active branch must return a
rank-2 array with the input row count. This explicitly enforces the family's
row-preservation contract; sklearn's concatenation only checks row agreement
among branch outputs. Input must also be rank 2; validation of finiteness,
nonempty training data and targets belongs to each estimator.

Weights multiply branch outputs, including passthrough, before concatenation.
Zero, negative, infinite and NaN weights are not implicitly removed or rejected.
Weight keys must be declared branch names, even for dropped branches. Unknown
keys, duplicate names, names containing `__`, and constructor-name collisions
raise before fitting. Configuration validation is eager at construction and
repeated before execution; sklearn defers some checks until fitting.

Passthrough contributes input columns. Drop contributes no columns and invokes
no estimator. Sentinels contain no estimator or learned array buffers. A
passthrough-only or all-dropped union is stateless for fitted checks. Other
unions delegate their fitted query to all active estimators; transform invokes
the child methods so stateless transformations remain possible before fit.

The output owns independent storage, including singleton and passthrough-only
unions. Strided input is accepted; no output aliases its input. Concatenation
uses the pinned NuMojo list operation; scalar weighting uses NuMojo arithmetic.
This is a correctness baseline, with no throughput or allocation-efficiency claim.

All-dropped output has shape `(n_samples, 0)` and the union's compile-time dtype.
**Mojo adaptation:** a Float32 union retains Float32 here, whereas the NumPy
reference defaults to Float64 for its all-dropped result. Active empty-column
outputs also retain the declared dtype. Zero rows/columns work where the child
transformers support them. The union forwards NaNs/infinities without imposing
its own finite-value restriction.

## Configuration, cloning and names

`get_params(deep=True)` returns `FeatureUnionParams` with all constructor options
and a typed recursive `transformer_list` record. `deep` is forwarded to each
estimator; the record shape remains fixed for `deep=False`. `set_params` validates
all proposed names and weight keys first, then updates child configuration in
order. It is not transactional on errors raised by a child. Returned records
and weight dictionaries own their values.

```mojo
var params = combined.get_params()
params.transformer_list.first.estimator = left_params
combined.set_params(params)
```

Access actual children through the typed tree, for example
`combined.transformer_list.first.estimator`. Replacing that field with a
same-type instance preserves the replacement's state. Replacing a branch with
a different estimator or sentinel type constructs a new union type. A sentinel's
`UnionSentinelParams.mode` can switch between passthrough and drop. Native
heterogeneous dictionary lookup and `step__parameter` keyword routing are not
implemented. `clone(combined)` recursively calls constituent clone hooks and
copies configuration, not learned buffers; stateless/frozen clone exceptions
retain their own semantics.

`get_feature_names_out(input_features=None)` prefixes branch output names when
`verbose_feature_names_out=True`. Without prefixes, duplicate output names
raise. Drop contributes no names. Passthrough uses explicit input names, or
`x0`, `x1`, ... after fit. With no active branches, the default prefixed form
returns an empty list; the unprefixed form raises, matching the reference.

Existing estimators opt into the shared `UnionFeatureNames` trait to
forward names; otherwise name requests raise an explicit AttributeError-style
error. Its method signature is:
`get_feature_names_out(self, input_features: Optional[List[String]] = None)
raises -> List[String]`.

`n_features_in_()` is a native method, delegating to the **first declared**
branch, not the first active one. Estimators opt into `UnionInputFeatures` with
`n_features_in_(self) raises -> Int`, or implement the shared
`InputFeatureCount.get_n_features_in` contract. `ColumnTransformer` supports
feature names and counts directly in a union. No duplicate feature-count state is stored
in the union. A first dropped branch, an unfitted passthrough, or an estimator
without this capability raises. The method form and optional capability traits
are explicit Mojo adaptations; a family may provide a thin forwarding adapter.
Dense arrays carry no string column metadata, so `feature_names_in_` is absent.

## Scope and evidence

Run `pixi run test` and `pixi run parity`. Native tests cover both dtypes,
fit-vs-fit_transform dispatch, optional targets, cloning, parameter ownership,
refits and failures, feature names/counts, recursive and nested unions, invalid
names/weights/shapes, zero dimensions, strided inputs, and output independence.
Negative compilation checks reject prediction, missing transform capability,
and incompatible dtypes. Both isolated consumer namespaces execute a union.
The runnable example and integration tests combine FeatureUnion, Pipeline and
the real DummyRegressor; numerical fixture transformers are explicitly test probes.

The parity driver executes FeatureUnion from local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, using the locked support wheel only
for surrounding utilities/base classes. It compares 50 matrix cases, 10 dispatch
and state cases, and output names. Float32 rtol/atol is `2e-6`, Float64 `1e-12`,
allowing scalar-weight rounding and decimal output conversion on bounded fixtures;
shapes and state are exact. NaN/Inf classifications must agree. The all-drop dtype
adaptation is checked explicitly. Source hashes/results are recorded under
ignored `builds/union-parity.json`.

Sparse inputs/outputs, mixed-dtype promotion, pandas/polars output, Python
reflection, dynamic heterogeneous indexing, arbitrary fit metadata and shared
metadata routing, serialization, and parallel scheduling remain unsupported.
Unsupported methods such as `set_output` and `get_metadata_routing` are absent;
extra arguments fail at compile time. `make_union` accepts two or three
transformers (see [factories](sequential.md#factories)). `named_transformers`
and `union[name]` lookup are absent; use the typed tree. This implementation does not claim complete sklearn parity.
