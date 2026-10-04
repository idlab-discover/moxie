# KernelRidge

`from moxie.kernel_ridge import KernelRidge` provides pure Mojo dense kernel
ridge regression over `numojo.NDArray`. X, y, and prediction inputs independently
accept Float32, Float64, Int32, or Int64. X is a nonempty finite matrix;
y is `(n,)` or `(n, targets)`. Predictions and `dual_coef_` preserve target
rank, including `(n, 1)`, and follow the reference promotion rules below.
`KernelRidge[dtype]` still selects the associated Input/Target types used by
generic estimator traits (Float64 by default); direct calls infer each input's
dtype independently. It does not force computation or output precision.

```mojo
import numojo as nm
from moxie.kernel_ridge import KernelRidge

var X = nm.arange[DType.float64](4).reshape(nm.Shape(4, 1))
var y = nm.arange[DType.float64](4)
var model = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
var prediction = model.fit(X, y).predict(X)
```

See the [family README](../README.md) for example and build commands.
Compile user programs with `pixi run mojo build -I builds/deps program.mojo`.

## Configuration and state

Constructor defaults are `alpha=1.0`, `kernel="linear"`, `gamma=None`,
`degree=3.0`, `coef0=1.0`, `kernel_params=None`, plus the native adaptation
`native_kernel=None`. Alpha is `ScalarOrVector`, a variant of Float64 and
`List[Float64]`: scalar, length one, or one finite nonnegative penalty per
target. Fit weights accept a scalar or one value per sample, finite and
nonnegative. Scalar weights and native lists carry Float64 values. See the
promotion rules below for their effect on intermediate and output dtypes.

Named kernels are linear, poly/polynomial, rbf, laplacian, sigmoid, cosine,
additive_chi2, chi2 and precomputed. Gamma defaults to 1 / feature count;
chi2 requires an explicit gamma because sklearn's estimator passes None to
its standalone kernel. Chi-squared features must be nonnegative. Gamma and
degree must be finite and nonnegative; coef0 must be finite. Named kernels
ignore `kernel_params`, matching sklearn.

Precomputed fit takes `(n, n)` and predict takes `(m, n)` (test versus training
samples). `n_features_in_` is n in this mode. Other modes record feature count.
`X_fit_`, `dual_coef_` and `n_features_in_` become valid after fitting.
Inference on an unfitted model raises a descriptive `NotFittedError` message.

`fit`, `set_params`, `set_fit_request`, and `set_score_request` return a
mutable borrow of the original estimator, supporting
`model.set_params(alpha=ScalarOrVector(2.0)).fit(X, y).predict(X)`. This includes
mapping, record, weighted, mixed-dtype, and feature-name overloads. A failed chain
does not undo earlier successful calls. `get_params(deep=True)` returns an
independently owned
`KernelRidge.ParameterMap`; `set_params(mapping)` validates all keys/types before
applying a partial update. Its `ParameterValue` variant represents supported
constructor values. Call `get_params_record()` for the legacy field-based
`KernelRidgeParams` record; `set_params(record)` replaces configuration.
Named updates are available for constructor parameters, including native_kernel.
Explicit `gamma=None`, `kernel_params=None`, and `native_kernel=None` reset
optional values; omitted keywords retain them. Updating parameters retains learned
state and prediction uses current configuration. `moxie.base.clone(model)` copies only
configuration; `model.copy()` also copies learned arrays. Successful refits
replace state; validation/solve failure preserves the prior fitted state.

Fit validates all constructor settings, including unused gamma, degree and
coef0. Prediction filters settings by the current kernel: for example, a
fitted linear model still predicts after `set_params(degree=-1.0)`, but a
subsequent fit rejects that degree. Alpha affects fitting only. Active kernel
settings affect prediction immediately without recomputing `dual_coef_`.
Native callable configuration conflicts remain errors in both fit and predict.

## Feature names and metadata requests

NuMojo arrays carry no column labels. Supply a `List[String]` using
`fit(X, y, feature_names=names)` and the same keyword on `predict` or `score`.
`feature_names_in_` is an optional, independently owned list: None before fit
or after a successful unnamed fit. Named inference checks exact names and
order, including duplicate names. Missing, renamed, reordered or extra labels
raise. Invalid name counts raise before fitting. A failed fit retains previous
learned state. Names supplied on only one side are accepted, as in sklearn;
Python's corresponding warnings are not emitted. No dataframe adapter is implied.

`set_fit_request(sample_weight=True/False/None)` and `set_score_request(...)`
configure requests independently. A String specifies an ASCII identifier alias;
omitting the keyword or using `"$UNCHANGED$"` retains its value. True requests
weights, False ignores them, and None rejects supplied weights. Direct fit/score
calls are unaffected. `get_metadata_routing()` returns an independent
`KernelRidgeMetadataRequest` with `fit` and `score` entries. Native parent code
can call `routing.fit.route(metadata)` on a `Dict[String, ScalarOrVector]` to
obtain optional weights; aliases select the corresponding key. Other metadata
keys are left to the parent to validate. This is a KernelRidge consumer contract,
not a global Pipeline router or sklearn configuration/context manager.
Requests survive clone and persistence; learned names survive fitted copies and
persistence, but are absent in an unfitted clone.

```mojo
var names: List[String] = ["height", "width"]
_ = model.fit(X, y, feature_names=names)
var prediction = model.predict(X, feature_names=names)
_ = model.set_fit_request(sample_weight=True)
_ = model.set_score_request(sample_weight=String("evaluation_weights"))
_ = model.set_params(gamma=None)
var params = model.get_params()
params["alpha"] = KernelRidge.ParameterValue(0.5)
_ = model.set_params(params)
```

## Native callable kernels

Import `RowKernel` from `moxie.kernel_ridge` and implement a Copyable struct with
`evaluate[dtype: DType](self, X: nm.NDArray[dtype], i: Int,
Y: nm.NDArray[dtype], j: Int) raises -> SIMD[dtype, 1]`.
Construct `KernelRidge[dtype, MyMetric](kernel=MyMetric(...))`. The older
`native_kernel=MyMetric(...)` spelling remains supported with the default kernel
name. `set_params(kernel=MyMetric(...))` replaces the callable;
`set_params(kernel="linear")` switches back to a named kernel. The active
callable is stored in `native_kernel`; the legacy `kernel` field remains a String.
`get_params()["kernel"]` exposes the active callable as an optional Metric arm.
Arbitrary typed state can live in the callable struct. To accept runtime
`kernel_params` keys with mixed values, import `KernelParameters` and
`KernelParameterValue` from `moxie.kernel_ridge`, then override
`evaluate_with_params[dtype: DType](self, X: nm.NDArray[dtype], i: Int,
Y: nm.NDArray[dtype], j: Int, params: Optional[KernelParameters])
raises -> SIMD[dtype, 1]`. `KernelParameters` is a
`Dict[String, KernelParameterValue]`; its value variant supports `Float64`,
`Int`, `Bool`, and `String`. Check `value.isa[T]()` before extracting `value[T]`,
and raise a descriptive error for wrong types or unknown keys in the callback.
For example:

```mojo
var params = KernelParameters()
params["scale"] = KernelParameterValue(Float64(1.5))
params["degree"] = KernelParameterValue(Int(2))
params["enabled"] = KernelParameterValue(True)
params["mode"] = KernelParameterValue(String("scaled"))
_ = model.set_params(kernel_params=params^)
```

Migrate existing `Dict[String, Float64]` declarations to `KernelParameters`
and extract numeric values with `[Float64]` in callbacks. This is a family API
change; the previous Float64-only restriction was an implementation choice.
The supported variant arms are deliberately bounded; additional structured
state can live in the typed callable struct. Parameters retain their types through
configuration copies, cloning, fitting, prediction, and persistence. The default
implementation accepts None/empty mappings and rejects nonempty mappings.
[Mojo supports typed `**kwargs`](https://mojolang.org/docs/manual/functions/#variadic-keyword-arguments);
this callback API passes an explicit typed map.
Python function objects and arbitrary Python objects are not supported. Rows
are borrowed and must be indexed using array strides. Same-data fit evaluates the upper triangle
and mirrors it, matching sklearn even for asymmetric callables. Prediction
evaluates every test/training pair. Tests show parameter cloning and dispatch.

## Numerical behavior and ownership

For D = diag(sqrt(weights)), solve `(D K D + alpha I) z = D y`, then
`dual_coef_ = D z`. Prediction uses the approved NuMojo rank-two matrix
product. Equal penalties share Cholesky; failure uses the numerical owner's
minimum-norm Jacobi SVD, with cutoff epsilon(dtype) times the largest singular
value. Unequal penalties have no fallback, matching sklearn. Cholesky reads
the upper triangle; fallback uses the full matrix. Zero weights, zero alpha,
singular, indefinite and nonsymmetric precomputed systems are supported.

Fit snapshots X and never mutates caller arrays. NuMojo strided snapshot copies
retain the complete backing allocation; solver workspaces materialize logical
entries. Prediction materializes coefficients as rank two and reshapes vector
outputs. Dtype conversion first materializes logical elements, so noncontiguous
and negative-stride views do not get cast by traversing physical storage. Storage is O(n² + n*d + n*t), factorization O(n³); prediction adds
O(m*n + m*t) storage. SVD may take multiple sweeps.

`score` returns uniformly averaged R², with optional finite scalar/vector
weights. Constant targets score 1 for perfect predictions and 0 otherwise;
valid weights with fewer than two samples return NaN. All-zero score weights
raise even for one sample; zero total score weight also raises. Signed score
weights are accepted, matching the reference metric. Score uses
Float64 accumulation to limit cancellation, including for Float32 inputs.

## Explicit language adaptations

NuMojo arrays have a compile-time dtype, but kernel and alpha settings can
change the result dtype at runtime. `FloatingArray` is the standard
`Variant[NDArray[float32], NDArray[float64]]`, used for `dual_coef_` and prediction
returns. `X_fit_` is `FittedArray`, a variant with Float32, Float64, Int32 and
Int64 arms; it retains the exact input dtype and independently owned values.
These aliases contain ordinary NuMojo arrays, with no new numerical container.
Inspect `.isa[nm.NDArray[DType.float64]]()` and borrow with the typed subscript,
or call `.copy()` on that arm for independent ownership:

```mojo
var X32 = nm.arange[DType.float32](4).reshape(nm.Shape(4, 1))
var y64 = nm.arange[DType.float64](4)
var mixed = KernelRidge()
_ = mixed.fit(X32, y64)
var result = mixed.predict(X32)
# This shared-alpha mixed fit returns Float64.
var values = result[nm.NDArray[DType.float64]].copy()
```

| Stage | Reference-matching dtype behavior |
| --- | --- |
| Named kernel, including precomputed | Float32 when both feature arrays are Float32; otherwise Float64. Integer features convert for computation. |
| Laplacian | Always Float64, including Float32 features. |
| Native callable | Evaluates rows and stores results in the common floating feature dtype, matching the pinned callable dispatcher's output dtype. |
| Targets | Float32 stays Float32; Float64 and integer targets use Float64 for solving. |
| No weights or scalar `1.0` | Leaves the target dtype unchanged. |
| Other scalar weights or native lists | Float64 square roots promote weighted targets to Float64; weighted K is rounded back to its kernel dtype. |
| Scalar, length-one, or equal per-target alpha | Solve/output uses the common dtype of K and weighted y. |
| Unequal per-target alpha | Solves in that common dtype, then stores each solution in K's dtype before final weight multiplication. No least-squares fallback. |
| Prediction | Kernel dtype combines with the stored coefficient dtype; Float64 in either promotes the product. |

Alpha equality is tested in the constructor's Float64 representation before
any kernel-dtype rounding. Regularization updates retain K's dtype before
solver promotion. Integer and mixed prediction inputs are supported, including
precomputed matrices. Each successful refit may change both fitted variant
arms. Post-fit parameter updates affect prediction kernels without converting
the stored coefficients.

The native weight API represents Float64 scalars/lists; NumPy Float32 weight
arrays have no separate native weight overload. Compare native lists with
Float64 reference arrays and scalar values with Python float weights.
Native callables receive rows converted to the common floating dtype; unlike
Python callbacks, they do not inspect the original integer/mixed row dtypes.
Int32 and Int64 are the supported integer input types; other integer widths,
Float16, complex, and boolean arrays are outside this milestone.

All-zero fit weights are supported in scalar and native-list form. The pinned
Python estimator accepts Python scalar `0.0` but rejects all-zero arrays through
its validator (including NumPy Float32 scalar zero). Native lists deliberately
retain the inherited engine's all-zero behavior; parity uses Python scalar zero
and compares its zero solution and reference-promoted dtype.

Malformed alpha lengths/values are rejected rather than reproducing accidental
private-solver broadcasting. Errors use Mojo `Error` with descriptive prefixes;
Python warning categories are not emitted. Native parameter mappings use a finite value variant rather than unrestricted
Python objects. Sparse input, Python callables, global sklearn metadata routing,
Python pickle serialization and pandas adapters are excluded. Sparse input is
deferred.

## Build and reference validation

The root Moxie `DEPENDENCIES.json` assembles current live sources, including
this family's modules. Run the build, test and parity commands in the
[family README](../README.md). Parity executes
the recorded local sklearn estimator, named and callable kernel bodies, solver,
and inherited score, using a support wheel for infrastructure. The local Cython
chi-squared helper is compiled for the oracle. Both dtypes cover coefficients,
predictions, shapes, scores, weighted/multioutput fits, and solver failures.
The dense dtype matrix checks input combinations, vector/column/multioutput
targets, named/callable/precomputed kernels, weights, scalar/equal/unequal alpha,
and rank-cutoff-sensitive fallback. It asserts output dtype arms as well as
values; raw reference records are in `builds/dtype-parity`. No kernel-output
cast is applied to the reference.
Float32 estimator tolerance is 2e-4; Float64 is 3e-9 (relative plus absolute).
Score permits five times these tolerances. Numerical helpers also compare
against SciPy least squares; ill-conditioned cases have explicit larger
condition-dependent tolerances. Raw fixtures, source hashes, and results live
under `builds/parity` and `builds/numerics-parity`. The original 520-case
Float64 fixture suite is retained against the new storage/estimator under
`builds/dense64-parity`.

Weighted score fixtures execute the pinned R² metric and its weight/target
validators. They cover both dtypes, vector and multioutput targets, constant
targets, signed and zero weights, singleton inputs, and invalid weights.
The score tolerance is 2e-5 for Float32 and 1e-12 for Float64 (relative plus
absolute), allowing the documented widened accumulation. Raw fixtures and
results are retained in `builds/parity/score-*`.

Post-fit mutation fixtures cover ignored parameters for every kernel, active
kernel changes, direct optional resets, feature names, request routing and parameter mappings.
Contract tests also verify that invalid ignored settings still fail on refit
and on fitting an unfitted clone.

RBF distances retain sklearn's staged arithmetic (`-2 * dot`, then each norm),
including its cancellation sensitivity for large, nearby values. Float32 RBF
distances accumulate in Float64 before casting, gamma scaling, and exp.

## Migrating from the initial dense prototype

The former List-backed `DenseArray` is replaced by the approved NuMojo
`NDArray[dtype]` storage. This is an explicit native-array API change: construct
arrays with NuMojo `Shape`/creation helpers, inspect `.shape`/`.ndim`, and use
NuMojo indexing instead of `.data`, `.rows`, and `.cols`. Default estimator dtype
remains Float64 for trait-associated inputs. Direct calls now accept mixed
dtypes. Coefficients, predictions and fitted inputs now expose typed variant
arms instead of one statically chosen array type; update array access as shown
above. Sparse input remains deferred.

## Native model persistence (experimental)

The family owns `KernelRidgeSchema[dtype, Metric, Adapter]` in
`moxie.kernel_ridge_serialization`. It uses Moxie's shared binary container and
`ModelSchema` contract; no fit runs during loading. The default schema dtype is
Float64, matching `KernelRidge`. Pass Float32 for `KernelRidge[DType.float32]`.

```mojo
from moxie.kernel_ridge_serialization import KernelRidgeSchema
from moxie.serialization import save_model, load_model

save_model[KernelRidgeSchema](model, "regressor.moxie")
var restored = load_model[KernelRidgeSchema]("regressor.moxie")
var predictions = restored.predict(X)
```

Build first with `pixi run build`, then compile against `-I builds/deps`.
The serialization engine and family schema are live Moxie modules assembled
through the root dependency manifest. The family persistence examples use raw payloads.

State schema 3 uses estimator ID `moxie.kernel_ridge.KernelRidge`. Its manifest
stores, in order: estimator dtype code; alpha vector flag and either a Float64
scalar or count-prefixed Float64 vector; kernel string; gamma presence and
optional Float64 value; Float64 degree and coef0; kernel_params presence and
optional count-prefixed entries (string key, unsigned type tag, typed value);
fitted flag; signed feature count; feature-name presence and optional count-prefixed strings; fit and score
sample-weight requests (presence flag, then Bool/string discriminator and value).
Parameter type tags are 0=Float64, 1=Int (signed 64-bit), 2=Bool, and
3=String. Unknown tags and malformed values are rejected. Schemas 1 and 2
remain readable, interpreting untagged parameters as Float64. Schema 1 defaults
new metadata to absent; saving writes schema 3. Invalid name counts and aliases
are rejected.
When a native kernel is present, the manifest then contains its stable
identifier string (1–4096 UTF-8 bytes), positive unsigned state version, and a
length-prefixed byte string containing adapter state. No suffix means no native
kernel. Integers,
flags, strings and scalars use the shared binary primitives.
Two named array payloads store `X_fit_` (Float32/Float64/Int32/Int64) and
`dual_coef_` (Float32/Float64), preserving their independent dtypes and target
rank. Strided arrays are serialized in logical C order and loaded into
independent contiguous arrays. An unfitted model has feature count zero and
two empty rank-one arrays of its estimator dtype.

Loading checks identity/version, estimator and payload dtypes, exact payload
names/count, manifest exhaustion, duplicate dictionary keys, fitted shapes,
sample and feature counts, and finite learned values. Encoding performs the
same checks. Current configuration is preserved separately from learned state,
including settings changed after fitting or settings that would fail on the
next fit. It is not revalidated as if it produced the stored arrays; ordinary
fit/predict validation still applies. In particular, current alpha length or
kernel choice does not constrain historical fitted shapes.

All named kernels and precomputed kernels are supported. Native callables require
an explicit `NativeKernelAdapter`, chosen as the third schema parameter:

```mojo
from callable_kernel import AffineKernel, AffineKernelAdapter

comptime Schema = KernelRidgeSchema[DType.float32, AffineKernel, AffineKernelAdapter]
save_model[Schema](model, "callable.moxie")
var restored = load_model[Schema]("callable.moxie")
```

See [the complete example](../examples/callable_kernel.mojo); compile it with
`-I builds/deps -I examples`. The adapter contract is:

- `Kernel`: the exact native kernel type, checked at compile time.
- `identifier() -> String` and `state_version() -> Int`: stable identity and
  positive state version. Change the version when the parameter format or its
  meaning changes; do not reuse identifiers for different kernels.
- `encode(kernel) -> List[UInt8]`: validate and encode owned parameter bytes.
- `decode(state: Span[UInt8, _]) -> Kernel`: validate all parameters and exact
  state length, then construct independent kernel configuration. Reject unknown
  fields, malformed values and unsupported state. Use the shared binary helpers.

The example encodes one Float64 offset, rejecting nonfinite values and surplus
bytes while preserving signed zero. Adapter code is trusted application code:
keep reconstruction data-only, without imports, executable lookup or dispatch
based on state bytes. The family loader checks identity and version before
calling the selected adapter. There is no runtime registry, and neither fit nor
kernel evaluation runs during loading. The default adapter rejects present
callable state. A typed model with an absent native kernel still uses the original
built-in layout. Earlier readers reject the new callable suffix.

No pickle compatibility, sparse persistence, schema migrations, or stable
cross-version format promise is provided. Shared load limits and corruption
checks apply; see
[Moxie's container contract](../../../docs/serialization.md).

`pixi run test` includes fitted/unfitted round trips, named/precomputed kernels,
vector/column/multiple targets, weights, mixed learned dtypes, strided snapshots,
exact large integers, post-fit parameter changes, file save/load, ownership,
clone/refit behavior, adapter identity/version/parameter rejection, and invalid
state rejection. Frozen [schema-v1 fixtures](../tests/fixtures/schema-v1/README.md)
check file compatibility and prediction bits without regenerating expected data.
Restored predictions and scores must equal the originals exactly; persistence
introduces no numerical conversion. Existing sklearn parity remains the algorithm
correctness check.
