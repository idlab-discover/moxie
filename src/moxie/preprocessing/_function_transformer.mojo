# Adapted from scikit-learn sklearn/preprocessing/_function_transformer.py
# (revision 4358542a1bd39af2e7e1cbace8b9fed067473c5b); authors: scikit-learn developers.
# The check_inverse comparison follows numpy.isclose (numpy/_core/numeric.py,
# BSD-3-Clause, NumPy developers). SPDX-License-Identifier: BSD-3-Clause
"""FunctionTransformer over typed native callables.

sklearn accepts arbitrary Python callables plus `kw_args` dictionaries. Natively
a callable is a value of a struct type conforming to `ArrayFunction`: the type
fixes the input/output dtypes and the type of its keyword-argument record, and
the instance may carry its own configuration. `Identity`, `Elementwise`,
`ArrayFn` and `KwArrayFn` adapt plain `def` functions. Feature-name callables
conform to `FeatureNamesCallable`. See docs/api/function.md.
"""
from moxie._feature_metadata import FeatureNamesOut as NativeFeatureNamesOut
from std.math import isfinite, isnan
from numojo import NDArray, Shape, zeros
from moxie.preprocessing._base import InvertiblePreprocessor
from moxie.preprocessing._common import check_set_output, check_n_features, dtype_name


comptime _NAME = "FunctionTransformer"
comptime _NOT_INVERSE = "The provided functions are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'."


# ---------------------------------------------------------------------------
# Typed callables


@fieldwise_init
struct NoKwArgs(ImplicitlyCopyable, Copyable, Movable):
    """The keyword-argument record of callables that take none (sklearn `{}`)."""
    pass


trait ArrayFunction(Copyable, Deinitable):
    """A typed `func` / `inverse_func`: `f(X, kw_args) -> X_out`.

    `InDType`/`OutDType` fix the array dtypes; `KwArgs` is the record type of
    the matching `kw_args` / `inv_kw_args` parameter (None when unset). The
    output may have any shape. Implementations must not retain `X`.
    """
    comptime InDType: DType
    comptime OutDType: DType
    comptime KwArgs: Copyable & Deinitable

    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]:
        ...


@fieldwise_init
struct Identity[dtype: DType](ArrayFunction, ImplicitlyCopyable):
    """sklearn `_identity` as an explicit callable: a compact copy of X."""
    comptime InDType = Self.dtype
    comptime OutDType = Self.dtype
    comptime KwArgs = NoKwArgs

    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]:
        return _compact[Self.dtype, Self.dtype](X)


struct Elementwise[i: DType, o: DType, //, f: def(Scalar[i]) thin -> Scalar[o]](ArrayFunction, ImplicitlyCopyable):
    """Apply a scalar `def` to every element (NumPy ufunc style); keeps the shape.

    `Elementwise[my_log1p]()` with `def my_log1p(x: Float64) -> Float64`.
    """
    comptime InDType = Self.i
    comptime OutDType = Self.o
    comptime KwArgs = NoKwArgs

    def __init__(out self):
        pass

    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]:
        var out = zeros[Self.o](_shape_of(X, 1))
        for k in range(out.size):
            out.unsafe_set(k, Self.f(X.unsafe_get(_offset(X, k, 1))))
        return out^


struct ArrayFn[i: DType, o: DType, //, f: def(NDArray[i]) thin raises -> NDArray[o]](ArrayFunction, ImplicitlyCopyable):
    """Adapt a whole-array `def(X) raises -> NDArray` taking no keyword arguments."""
    comptime InDType = Self.i
    comptime OutDType = Self.o
    comptime KwArgs = NoKwArgs

    def __init__(out self):
        pass

    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]:
        return Self.f(X)


struct KwArrayFn[i: DType, o: DType, K: Copyable & Deinitable, //, f: def(NDArray[i], Optional[K]) thin raises -> NDArray[o]](ArrayFunction, ImplicitlyCopyable):
    """Adapt a `def(X, kw_args: Optional[K]) raises -> NDArray`; K is the kw_args record."""
    comptime InDType = Self.i
    comptime OutDType = Self.o
    comptime KwArgs = Self.K

    def __init__(out self):
        pass

    def __call__(self, X: NDArray[Self.InDType], kw_args: Optional[Self.KwArgs]) raises -> NDArray[Self.OutDType]:
        return Self.f(X, kw_args)


trait FeatureNamesCallable(Copyable, Deinitable):
    """A typed `feature_names_out` callable.

    Receives the validated input feature names (None when the transformer is
    unfitted and none were supplied), like sklearn's second argument. The
    transformer itself is not passed: configuration the names depend on lives
    in the callable's own fields.
    """
    def __call__(self, input_features: Optional[List[String]]) raises -> List[String]:
        ...


struct NamesFn[f: def(Optional[List[String]]) thin raises -> List[String]](FeatureNamesCallable, ImplicitlyCopyable):
    """Adapt a plain `def(input_features) raises -> List[String]`."""
    def __init__(out self):
        pass

    def __call__(self, input_features: Optional[List[String]]) raises -> List[String]:
        return Self.f(input_features)


@fieldwise_init
struct NoNamesCallable(FeatureNamesCallable, ImplicitlyCopyable):
    """Default names-callable type when `feature_names_out` is None or 'one-to-one'."""
    def __call__(self, input_features: Optional[List[String]]) raises -> List[String]:
        raise Error("TypeError: no feature_names_out callable was supplied")


struct FeatureNamesOut[N: FeatureNamesCallable](Copyable, Movable):
    """The sklearn `feature_names_out` value: None, a string ('one-to-one') or a callable.

    Converts implicitly from None, a string or an N instance. Strings are kept
    verbatim and validated at fit, as sklearn's parameter validation does.
    """
    var _kind: Int  # 0 None, 1 string, 2 callable, -1 "not given" (set_params only)
    var _text: String
    var _callable: Optional[Self.N]

    @implicit
    def __init__(out self, none: NoneType._mlir_type):
        self._kind = 0
        self._text = ""
        self._callable = None

    @implicit
    def __init__(out self, text: StringLiteral):
        self._kind = 1
        self._text = String(text)
        self._callable = None

    @implicit
    def __init__(out self, text: String):
        self._kind = 1
        self._text = text
        self._callable = None

    @implicit
    def __init__(out self, callable: Self.N):
        self._kind = 2
        self._text = ""
        self._callable = callable.copy()

    @staticmethod
    def _not_given() -> Self:
        var value = Self(None)
        value._kind = -1
        return value^

    def is_none(self) -> Bool:
        return self._kind == 0

    def is_one_to_one(self) -> Bool:
        return self._kind == 1 and self._text == "one-to-one"

    def is_callable(self) -> Bool:
        return self._kind == 2

    def text(self) -> String:
        """The string value ('' unless a string was given)."""
        return self._text

    def callable(self) raises -> Self.N:
        if self._kind != 2:
            raise Error("ValueError: feature_names_out is not a callable")
        return self._callable.value().copy()

    def _repr(self) -> String:
        if self._kind == 0:
            return "None"
        if self._kind == 1:
            return "'" + self._text + "'"
        return "<callable>"

    def _validate(self) raises:
        if self._kind == 1 and self._text != "one-to-one":
            raise Error("ValueError: The 'feature_names_out' parameter of FunctionTransformer must be a callable, a str among {'one-to-one'} or None. Got '" + self._text + "' instead.")


# ---------------------------------------------------------------------------
# Array helpers (arbitrary rank, strided, offset views)


def _shape_of[d: DType](x: NDArray[d], row_step: Int) raises -> Shape:
    """x's shape with the first axis subsampled by `row_step` (X[::row_step])."""
    var dims = List[Int]()
    for axis in range(x.ndim):
        var n = x.shape[axis]
        dims.append((n + row_step - 1) // row_step if axis == 0 else n)
    return Shape(dims)


@always_inline
def _offset[d: DType](x: NDArray[d], flat: Int, row_step: Int) -> Int:
    """Storage offset of the C-order logical element `flat` of X[::row_step]."""
    var rest = flat
    var offset = 0
    var axis = x.ndim - 1
    while axis > 0:
        var n = Int(x.shape.unsafe_load(axis))
        offset += (rest % n) * Int(x.strides.unsafe_load(axis))
        rest //= n
        axis -= 1
    if x.ndim > 0:
        offset += rest * row_step * Int(x.strides.unsafe_load(0))
    return offset


def _compact[i: DType, o: DType](x: NDArray[i], row_step: Int = 1) raises -> NDArray[o]:
    """Compact C-order copy of X[::row_step] (only the logical elements)."""
    var out = zeros[o](_shape_of(x, row_step))
    for k in range(out.size):
        out.unsafe_set(k, x.unsafe_get(_offset(x, k, row_step)).cast[o]())
    return out^


def _shape_text[d: DType](x: NDArray[d]) -> String:
    var text = String("(")
    for axis in range(x.ndim):
        if axis > 0:
            text += ", "
        text += String(Int(x.shape.unsafe_load(axis)))
    if x.ndim == 1:
        text += ","
    return text + ")"


def _check_array[d: DType](x: NDArray[d], *, estimator: Bool) raises:
    """sklearn check_array(X) for dense numeric input (validate=True).

    `estimator` selects the messages of validate_data (fit/transform) versus a
    bare check_array (inverse_transform).
    """
    var context = String(" by ") + _NAME if estimator else String("")
    if x.ndim < 2:
        raise Error("ValueError: Expected 2D array, got " + ("scalar" if x.ndim == 0 else "1D") + " array instead: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.")
    if x.ndim > 2:
        raise Error("ValueError: Found array with dim " + String(x.ndim) + ", while dim <= 2 is required" + context + ".")
    comptime if d.is_floating_point():
        for k in range(x.size):
            var value = x.unsafe_get(_offset(x, k, 1))
            if not isfinite(value):
                var input = String("Input X") if estimator else String("Input")
                if isnan(value):
                    var message = "ValueError: " + input + " contains NaN."
                    if estimator:
                        message += " " + _NAME + " does not accept missing values encoded as NaN natively."
                    raise Error(message)
                raise Error("ValueError: " + input + " contains infinity or a value too large for dtype('" + dtype_name[d]() + "').")
    if x.shape[0] < 1:
        raise Error("ValueError: Found array with 0 sample(s) (shape=" + _shape_text(x) + ") while a minimum of 1 is required" + context + ".")
    if x.shape[1] < 1:
        raise Error("ValueError: Found array with 0 feature(s) (shape=" + _shape_text(x) + ") while a minimum of 1 is required" + context + ".")


def _allclose[d: DType](x: NDArray[d], y: NDArray[d]) raises -> Bool:
    """numpy.allclose(x, y, rtol=1e-7, atol=1e-9) with broadcasting, equal_nan=False.

    Floating inputs compare in their own precision (NumPy weak-scalar
    promotion); integers compare in Float64, as numpy.isclose does.
    """
    var ndim = max(x.ndim, y.ndim)
    var dims = List[Int]()
    for k in range(ndim):
        var a = x.shape[x.ndim - ndim + k] if x.ndim - ndim + k >= 0 else 1
        var b = y.shape[y.ndim - ndim + k] if y.ndim - ndim + k >= 0 else 1
        if a != b and a != 1 and b != 1:
            raise Error("ValueError: operands could not be broadcast together with shapes " + _shape_text(x) + " " + _shape_text(y))
        dims.append(b if a == 1 else a)
    var total = 1
    for k in range(ndim):
        total *= dims[k]
    var index = List[Int](length=ndim, fill=0)
    for _ in range(total):
        var ox = 0
        var oy = 0
        for k in range(ndim):
            var ax = x.ndim - ndim + k
            var ay = y.ndim - ndim + k
            if ax >= 0 and x.shape[ax] != 1:
                ox += index[k] * Int(x.strides[ax])
            if ay >= 0 and y.shape[ay] != 1:
                oy += index[k] * Int(y.strides[ay])
        if not _isclose(x.unsafe_get(ox), y.unsafe_get(oy)):
            return False
        var k = ndim - 1
        while k >= 0:
            index[k] += 1
            if index[k] < dims[k]:
                break
            index[k] = 0
            k -= 1
    return True


@always_inline
def _isclose[d: DType](a: Scalar[d], b: Scalar[d]) -> Bool:
    comptime if d.is_floating_point():
        var close = abs(a - b) <= Scalar[d](1e-9) + Scalar[d](1e-7) * abs(b)
        return (close and isfinite(b)) or a == b
    else:
        var fa = a.cast[DType.float64]()
        var fb = b.cast[DType.float64]()
        return (abs(fa - fb) <= 1e-9 + 1e-7 * abs(fb)) or fa == fb


def _check_feature_names_in(
    n_features_in: Optional[Int], feature_names_in: Optional[List[String]],
    input_features: Optional[List[String]],
) raises -> List[String]:
    """sklearn `_check_feature_names_in` with an optional n_features_in_."""
    if input_features:
        var supplied = input_features.value().copy()
        if feature_names_in and supplied != feature_names_in.value():
            raise Error("ValueError: input_features is not equal to feature_names_in_")
        if n_features_in and len(supplied) != n_features_in.value():
            raise Error("ValueError: input_features should have length equal to number of features (" + String(n_features_in.value()) + "), got " + String(len(supplied)))
        return supplied^
    if feature_names_in:
        return feature_names_in.value().copy()
    if not n_features_in:
        raise Error("ValueError: Unable to generate feature names without n_features_in_")
    var generated = List[String]()
    for k in range(n_features_in.value()):
        generated.append("x" + String(k))
    return generated^


# ---------------------------------------------------------------------------
# FunctionTransformer


struct FunctionTransformerParams[F: ArrayFunction, G: ArrayFunction, N: FeatureNamesCallable](Copyable, Movable):
    """Constructor configuration of FunctionTransformer[F, G, N]."""
    var func: Optional[Self.F]
    var inverse_func: Optional[Self.G]
    var validate: Bool
    var accept_sparse: Bool
    var check_inverse: Bool
    var feature_names_out: FeatureNamesOut[Self.N]
    var kw_args: Optional[Self.F.KwArgs]
    var inv_kw_args: Optional[Self.G.KwArgs]

    def __init__(
        out self, func: Optional[Self.F] = None, inverse_func: Optional[Self.G] = None, *,
        validate: Bool = False, accept_sparse: Bool = False, check_inverse: Bool = True,
        feature_names_out: FeatureNamesOut[Self.N] = None,
        kw_args: Optional[Self.F.KwArgs] = None, inv_kw_args: Optional[Self.G.KwArgs] = None,
    ):
        self.func = func.copy()
        self.inverse_func = inverse_func.copy()
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out.copy()
        self.kw_args = kw_args.copy()
        self.inv_kw_args = inv_kw_args.copy()


struct FunctionTransformer[
    F: ArrayFunction = Identity[DType.float64],
    G: ArrayFunction = Identity[F.OutDType],
    N: FeatureNamesCallable = NoNamesCallable,
](InvertiblePreprocessor, Copyable, NativeFeatureNamesOut):
    """Construct a transformer from a typed callable; stateless (transform works unfitted).

    `func=None` / `inverse_func=None` select the identity. F's dtypes fix
    `Input` (NDArray[F.InDType]) and `Transformed` (NDArray[F.OutDType]); G
    must accept F.OutDType. `inverse_check_warning_` holds sklearn's
    UserWarning text when the check_inverse round trip failed at the last fit.
    """
    comptime Params = FunctionTransformerParams[Self.F, Self.G, Self.N]
    comptime Input = NDArray[Self.F.InDType]
    comptime Target = NDArray[Self.F.InDType]
    comptime Transformed = NDArray[Self.F.OutDType]
    var func: Optional[Self.F]
    var inverse_func: Optional[Self.G]
    var validate: Bool
    var accept_sparse: Bool
    var check_inverse: Bool
    var feature_names_out: FeatureNamesOut[Self.N]
    var kw_args: Optional[Self.F.KwArgs]
    var inv_kw_args: Optional[Self.G.KwArgs]
    var n_features_in_: Optional[Int]
    var feature_names_in_: Optional[List[String]]
    var inverse_check_warning_: Optional[String]

    def __init__(out self, func: Optional[Self.F] = None, inverse_func: Optional[Self.G] = None, *, validate: Bool = False, accept_sparse: Bool = False, check_inverse: Bool = True, feature_names_out: FeatureNamesOut[Self.N] = None, kw_args: Optional[Self.F.KwArgs] = None, inv_kw_args: Optional[Self.G.KwArgs] = None):
        comptime assert Self.G.InDType == Self.F.OutDType, "inverse_func must accept the dtype that func returns"
        self.func = func.copy()
        self.inverse_func = inverse_func.copy()
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out.copy()
        self.kw_args = kw_args.copy()
        self.inv_kw_args = inv_kw_args.copy()
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.inverse_check_warning_ = None

    def __init__(out self, func: Optional[Self.F] = None, inverse_func: Optional[Self.G] = None, *, validate: Bool = False, accept_sparse: Bool = False, check_inverse: Bool = True, feature_names_out: Self.N, kw_args: Optional[Self.F.KwArgs] = None, inv_kw_args: Optional[Self.G.KwArgs] = None):
        """Overload that infers N from a `feature_names_out` callable."""
        self = Self(func, inverse_func, validate=validate, accept_sparse=accept_sparse, check_inverse=check_inverse, feature_names_out=FeatureNamesOut[Self.N](feature_names_out), kw_args=kw_args, inv_kw_args=inv_kw_args)

    # -- configuration -------------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.func, self.inverse_func, validate=self.validate, accept_sparse=self.accept_sparse, check_inverse=self.check_inverse, feature_names_out=self.feature_names_out, kw_args=self.kw_args, inv_kw_args=self.inv_kw_args)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.func = params.func.copy()
        self.inverse_func = params.inverse_func.copy()
        self.validate = params.validate
        self.accept_sparse = params.accept_sparse
        self.check_inverse = params.check_inverse
        self.feature_names_out = params.feature_names_out.copy()
        self.kw_args = params.kw_args.copy()
        self.inv_kw_args = params.inv_kw_args.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        func: Optional[Self.F] = None,
        inverse_func: Optional[Self.G] = None,
        validate: Optional[Bool] = None,
        accept_sparse: Optional[Bool] = None,
        check_inverse: Optional[Bool] = None,
        feature_names_out: FeatureNamesOut[Self.N] = FeatureNamesOut[Self.N]._not_given(),
        kw_args: Optional[Self.F.KwArgs] = None,
        inv_kw_args: Optional[Self.G.KwArgs] = None,
    ) raises -> ref[self] Self:
        """Update the given parameters. An omitted Optional leaves the value unchanged;
        reset func/inverse_func/kw_args to None with `set_params(record)`.
        `feature_names_out=None` does reset it."""
        if func:
            self.func = func.copy()
        if inverse_func:
            self.inverse_func = inverse_func.copy()
        if validate:
            self.validate = validate.value()
        if accept_sparse:
            self.accept_sparse = accept_sparse.value()
        if check_inverse:
            self.check_inverse = check_inverse.value()
        if feature_names_out._kind != -1:
            self.feature_names_out = feature_names_out.copy()
        if kw_args:
            self.kw_args = kw_args.copy()
        if inv_kw_args:
            self.inv_kw_args = inv_kw_args.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.func, self.inverse_func, validate=self.validate, accept_sparse=self.accept_sparse, check_inverse=self.check_inverse, feature_names_out=self.feature_names_out, kw_args=self.kw_args, inv_kw_args=self.inv_kw_args)

    def __sklearn_is_fitted__(self) -> Bool:
        """Stateless: always True, as in sklearn."""
        return True

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting -------------------------------------------------------------

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        """Record n_features_in_ (rank >= 2 input), validate X if `validate`,
        and run the check_inverse round trip when both callables are given."""
        self._fit(X, None)
        return self

    def fit_named[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        """`fit` on a named table: records feature_names_in_ (X must be rank 2)."""
        if X.ndim != 2:
            raise Error("ValueError: fit_named requires a rank-2 X, got rank " + String(X.ndim))
        if len(feature_names) != X.shape[1]:
            raise Error("ValueError: feature_names length " + String(len(feature_names)) + " does not match n_features " + String(X.shape[1]))
        self._fit(X, feature_names.copy())
        return self

    def _fit(mut self, X: Self.Input, var names: Optional[List[String]]) raises:
        self.feature_names_out._validate()
        if self.validate:
            _check_array(X, estimator=True)
        self.feature_names_in_ = names^
        # sklearn `_check_n_features(reset=True)` skips (keeps the old value)
        # when X has no second axis.
        if X.ndim >= 2:
            self.n_features_in_ = X.shape[1]
        self.inverse_check_warning_ = None
        if self.check_inverse and self.func and self.inverse_func:
            self._check_inverse_transform(X)

    def _check_inverse_transform(mut self, X: Self.Input) raises:
        if X.ndim == 0:
            raise Error("IndexError: too many indices for array: array is 0-dimensional")
        var step = max(1, X.shape[0] // 100)
        if step == 1:
            self._round_trip(X)
        else:
            self._round_trip(_compact[Self.F.InDType, Self.F.InDType](X, step))

    def _round_trip(mut self, sample: Self.Input) raises:
        var round_trip = self.inverse_transform(self.transform(sample))
        comptime if Self.F.InDType == DType.bool:
            raise Error("ValueError: 'check_inverse' is only supported when all the elements in `X` is numerical.")
        if not _allclose(sample, round_trip):
            self.inverse_check_warning_ = String(_NOT_INVERSE)

    # -- transforms ----------------------------------------------------------

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if self.validate:
            _check_array(X, estimator=True)
            if self.n_features_in_:
                check_n_features(_NAME, self.n_features_in_.value(), X.shape[1])
        if self.func:
            return self.func.value()(X, self.kw_args)
        if self.kw_args:
            raise Error("TypeError: _identity() got an unexpected keyword argument (kw_args is set but func is None)")
        comptime if Self.F.InDType == Self.F.OutDType:
            return _compact[Self.F.InDType, Self.F.OutDType](X)
        else:
            raise Error("TypeError: func=None is the identity, which cannot produce " + dtype_name[Self.F.OutDType]() + " output from " + dtype_name[Self.F.InDType]() + " input")

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        if self.validate:
            _check_array(X, estimator=False)
        if self.inverse_func:
            var out = self.inverse_func.value()(rebind[NDArray[Self.G.InDType]](X), self.inv_kw_args)
            comptime if Self.G.OutDType == Self.F.InDType:
                return rebind_var[NDArray[Self.F.InDType]](out^)
            else:
                raise Error("TypeError: inverse_func returns " + dtype_name[Self.G.OutDType]() + " but this transformer's input dtype is " + dtype_name[Self.F.InDType]())
        if self.inv_kw_args:
            raise Error("TypeError: _identity() got an unexpected keyword argument (inv_kw_args is set but inverse_func is None)")
        comptime if Self.F.InDType == Self.F.OutDType:
            return _compact[Self.F.OutDType, Self.F.InDType](X)
        else:
            raise Error("TypeError: inverse_func=None is the identity, which cannot produce " + dtype_name[Self.F.InDType]() + " output from " + dtype_name[Self.F.OutDType]() + " input")

    # -- feature names -------------------------------------------------------

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """Defined only when feature_names_out is not None (else AttributeError)."""
        if self.feature_names_out.is_none():
            raise Error("AttributeError: This 'FunctionTransformer' has no attribute 'get_feature_names_out'")
        var names: Optional[List[String]] = None
        if self.n_features_in_ or input_features:
            names = _check_feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        if self.feature_names_out.is_one_to_one():
            if not names:
                # sklearn returns np.asarray(None, dtype=object) here.
                raise Error("ValueError: Unable to generate feature names without n_features_in_")
            return names.value().copy()
        if self.feature_names_out.is_callable():
            return self.feature_names_out._callable.value()(names)
        raise Error("ValueError: feature_names_out=" + self.feature_names_out._repr() + " is invalid. It must either be \"one-to-one\" or a callable with two arguments: the function transformer and an array-like of input feature names. The callable must return an array-like of output feature names.")
