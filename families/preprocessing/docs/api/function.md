# FunctionTransformer

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_function_transformer.mojo`.

| Name | Native signature |
| --- | --- |
| `FunctionTransformer[F, G, N]` | `FunctionTransformer(func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None)` |

Type parameters: `F: ArrayFunction = Identity[DType.float64]` is the type of
`func`; `G: ArrayFunction = Identity[F.OutDType]` is the type of
`inverse_func`; `N: FeatureNamesCallable = NoNamesCallable` is the type of a
`feature_names_out` callable. Mojo infers all three from the constructor
arguments. A second constructor overload takes `feature_names_out` as an `N`
value so that N can also be inferred.
`Input = NDArray[F.InDType]` and `Transformed = NDArray[F.OutDType]`. The struct
conforms to `InvertiblePreprocessor`, so it can be a Pipeline step.

## Typed callables

sklearn accepts any Python callable plus `kw_args` dictionaries. In Moxie,
`func` and `inverse_func` are values of a struct type that conforms to
`ArrayFunction`. The callable types below are imported from
`moxie.preprocessing._function_transformer`:

```mojo
trait ArrayFunction(Copyable, Deinitable):
    comptime InDType: DType
    comptime OutDType: DType
    comptime KwArgs: Copyable & Deinitable          # the type of kw_args
    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]
```

| Adapter | Wraps |
| --- | --- |
| `Identity[dtype]()` | sklearn's `_identity`: returns a compact copy of X |
| `Elementwise[f]()` | `def f(x: Scalar[i]) -> Scalar[o]` applied to every element, like a NumPy ufunc; the shape is kept |
| `ArrayFn[f]()` | `def f(X: NDArray[i]) raises -> NDArray[o]`, a whole-array function with any output shape |
| `KwArrayFn[f]()` | `def f(X: NDArray[i], kw: Optional[K]) raises -> NDArray[o]`; `K` becomes the type of `kw_args` |
| `NamesFn[f]()` | `def f(input_features: Optional[List[String]]) raises -> List[String]` used as a `feature_names_out` callable |

The adapters take functions as compile-time parameters (`thin` function
values), so they cost no storage and can be copied. A user struct can implement
`ArrayFunction` directly when a callable needs configuration in its own fields.

```mojo
from std.math import sqrt
from moxie.preprocessing import FunctionTransformer
from moxie.preprocessing._function_transformer import Elementwise, KwArrayFn, NamesFn

def root(x: Float64) -> Float64:
    return sqrt(x)

def square(x: Float64) -> Float64:
    return x * x

@fieldwise_init
struct Affine(Copyable, Movable):
    var factor: Float64
    var shift: Float64

def affine(X: NDArray[DType.float64], kw: Optional[Affine]) raises -> NDArray[DType.float64]:
    ...                                     # X * factor + shift

var ft = FunctionTransformer(Elementwise[root](), Elementwise[square](), validate=True)
var Xt = ft.fit_transform(X)                # check_inverse runs inside fit
if ft.inverse_check_warning_:               # sklearn would emit a UserWarning
    print(ft.inverse_check_warning_.value())
var scaled = FunctionTransformer(KwArrayFn[affine](), kw_args=Affine(2.0, 1.0))
scaled.set_params(kw_args=Affine(3.0, 0.0))  # like sklearn's dict update
var named = FunctionTransformer(feature_names_out="one-to-one")
```

## Behavior

Behavior follows sklearn:

* The transformer is stateless. `transform` and `inverse_transform` work before
  `fit`, and `__sklearn_is_fitted__()` is always True.
* `fit` sets `n_features_in_` from `X.shape[1]` for input of rank 2 or more,
  whether or not `validate` is set. Like sklearn, a rank-1 `X` leaves the
  previous value in place. `n_features_in_` is `Optional[Int]` because sklearn
  sometimes leaves the attribute undefined. `fit_named` also records
  `feature_names_in_`, and an ordinary `fit` clears it.
* `validate=False`: no checks are made. Input of any rank, NaN and infinity
  reach `func` unchanged, and `transform` does not check the feature count.
* `validate=True`: `fit` and `transform` apply sklearn's `check_array`. The
  input must have rank 2, at least one sample and one feature, and contain no
  NaN or infinity (integer and bool input are accepted). After `fit`,
  `transform` also checks the feature count. `inverse_transform` applies only
  the bare check, with no feature count and with sklearn's
  "Input contains NaN." wording.
* `func=None` and `inverse_func=None` are the identity, which returns a
  compact copy because the borrowed input cannot be returned. If `kw_args` or
  `inv_kw_args` is set while the matching function is None, the call raises
  `TypeError`, as sklearn's `_identity(X, **kw)` does.
* `check_inverse=True` with both functions set: `fit` runs
  `inverse_transform(transform(X[::max(1, n // 100)]))` and compares the result
  with `numpy.allclose(rtol=1e-7, atol=1e-9)` semantics. The comparison uses
  broadcasting, treats NaN as not close, and runs in the input precision
  (integers in Float64). Bool input raises sklearn's "only supported when all
  the elements in `X` is numerical" `ValueError`.
* `get_feature_names_out(input_features=None)`: with `feature_names_out=None`
  it raises `AttributeError`, as sklearn's `available_if` does. With
  `"one-to-one"` it returns the validated input names, or `x0..` when fitted
  without names. A callable receives the validated names, or None when the
  transformer is unfitted and no names are given. An invalid string raises
  sklearn's parameter-validation `ValueError` at `fit`, and the "is invalid"
  `ValueError` from `get_feature_names_out`.
* `accept_sparse` is stored and has no effect, because sparse input does not
  exist by type (see [scope](../scope.md)). `set_output` accepts only None and
  "default".

## Native adaptations

* **Typed callables.** A function's dtypes and keyword-argument record are part
  of its type, and `kw_args` is an `Optional[F.KwArgs]` record instead of a
  dictionary. sklearn's `kw_args={}` matches `kw_args=None`. A record that is
  present always counts as given, even an empty one, so `func=None` with any
  `kw_args` raises `TypeError`.
* **Output dtype.** `F.OutDType` may differ from `F.InDType`; for example
  `Elementwise[to_f32]` turns Float64 into Float32. The output shape is free.
  `inverse_func` must accept `F.OutDType`; this is checked at compile time.
  `inverse_transform` returns `NDArray[F.InDType]`. An identity inverse, or an
  `inverse_func` that returns another dtype, raises `TypeError` there. sklearn
  would return the array unchanged, but the trait fixes the return type.
* **Warnings.** Moxie has no warning channel. When the check_inverse round trip
  fails, `fit` completes like sklearn and stores sklearn's exact UserWarning
  text in `inverse_check_warning_: Optional[String]`. Every fit resets it to
  None, so the result is never silently reported as a success.
* **feature_names_out callables** receive only `input_features`. sklearn also
  passes the transformer, usually to read `kw_args`. A Moxie names callable
  keeps such configuration in its own fields (`AppendNames(count)` in the
  tests). The value is a `FeatureNamesOut[N]`, which converts implicitly from
  None, a string or an `N` value.
* **Unfitted "one-to-one"** with no `input_features` raises `ValueError`.
  sklearn returns the meaningless `np.asarray(None, dtype=object)` in that case.
* **set_params.** The keyword overload changes only the parameters it is given.
  `feature_names_out=None` resets that parameter. To reset `func`,
  `inverse_func`, `kw_args` or `inv_kw_args` to None, pass a modified
  `get_params()` record to `set_params(record)`.
* Callables must be copyable. Capturing closures are not copyable in Mojo 1.0,
  so they cannot be stored. Use a struct that implements `ArrayFunction` with
  fields instead. `get_metadata_routing` and pandas/polars output are
  unsupported.

## Memory

`transform` allocates only what the callable returns; the identity and
`Elementwise` allocate exactly one compact result. The check_inverse subsample
is materialized only when `n >= 200`. It then has `ceil(n / (n // 100))` rows,
at most 199, plus the callables' outputs for those rows. When `n < 200` the
borrowed X is used directly, with no copy. Strided and negative-stride views
are read element by element; no backing-buffer copy is made.

## Parity

`tests/parity/test_function.py`, 159 checks. Each native callable is paired
with the NumPy callable sklearn receives:

* identity with `validate` False/True on Float64 and Float32, including
  negative-stride and column-major views, and NaN/inf/rank-1 passthrough
* sqrt/around, x·(1+δ) for δ = 5e-8, 1e-7 and 2e-7 in both dtypes (both sides
  of the allclose boundary), and log1p/expm1
* an inverse that corrupts only odd rows, for n = 99…1001 (the subsample)
* affine `kw_args` with `set_params` update, refit and reset; column
  add/drop; Float64→Float32 cast; Int64 doubling; bool negation
* feature names (generated, given, callable, unvalidated)
* 12 error checks: validation, feature-count, parameter, identity
  `kw_args`, bool check_inverse and names errors

Outputs, inverse outputs, `n_features_in_`, warning text and names are
compared exactly. The one exception is log1p/expm1 output, compared with
rtol 1e-9 (Float64) and 2e-6 (Float32). Mojo 1.0's `std.math.log1p` differs
from NumPy's libm by up to about 3e-10 relative in Float64. That is a property
of the test callable; the transformer forwards callable output unchanged.
Errors match sklearn's exception class and message prefix.
