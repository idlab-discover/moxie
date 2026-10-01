# Adapted from scikit-learn sklearn/compose/_target.py (TransformedTargetRegressor)
# and sklearn/preprocessing/_function_transformer.py (FunctionTransformer with
# validate=True); authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Regression on a transformed target (`TransformedTargetRegressor`)."""
from numojo import NDArray, Shape, zeros
from std.math import isnan, isinf, nan
from std.sys import stderr
from moxie.base import (
    BaseEstimator,
    InverseTransformer,
    RegressorMixin,
    UnsupervisedEstimator,
    WeightedSupervisedEstimator,
    clone,
)
from moxie.utils.validation import check_is_fitted
from moxie.compose._base import TransformedTargetRegressorAPI
from moxie.compose._table import ColumnTable
from moxie.linear_model import LinearRegression


comptime TargetFunction[dtype: DType] = def(NDArray[dtype]) thin raises -> NDArray[dtype]
"""A target function or its inverse: receives and returns a dense array."""

comptime TargetRegressor = RegressorMixin & Copyable & Deinitable
"""Regressors usable by TransformedTargetRegressor (Target/Prediction NDArray[dtype])."""

comptime TargetTransformer = InverseTransformer & UnsupervisedEstimator & Copyable & Deinitable
"""Target transformers (Input/Transformed NDArray[dtype], fit on a 2-D target)."""


# ---------------------------------------------------------------------------
# Dense helpers. Inputs may be strided views: every access uses the element
# displacement implied by NuMojo strides (NuMojo applies the view offset).


def _shape_text[dtype: DType](a: NDArray[dtype]) raises -> String:
    """Python tuple spelling of a shape, as in sklearn messages: (3,), (3, 1)."""
    if a.ndim == 1:
        return String("(", a.shape[0], ",)")
    var text = String("(")
    for i in range(a.ndim):
        if i > 0:
            text += ", "
        text += String(a.shape[i])
    return text + ")"


def _displacement[dtype: DType](a: NDArray[dtype], flat: Int) raises -> Int:
    """Element displacement of the C-order position `flat`."""
    var rest = flat
    var offset = 0
    for axis in range(a.ndim - 1, -1, -1):
        var extent = a.shape[axis]
        offset += (rest % extent) * Int(a.strides[axis])
        rest //= extent
    return offset


def _value[dtype: DType](a: NDArray[dtype], flat: Int) raises -> Scalar[dtype]:
    return a.unsafe_get(_displacement(a, flat))


def _check_finite[dtype: DType](a: NDArray[dtype], input_name: String) raises:
    """Equivalent of sklearn `_assert_all_finite`: NaN takes precedence over infinity."""
    var subject = String("Input ", input_name, " ") if input_name.byte_length() > 0 else String("Input ")
    var has_inf = False
    for i in range(a.size):
        var v = _value(a, i)
        if isnan(v):
            raise Error(String("ValueError: ", subject, "contains NaN."))
        has_inf = has_inf or isinf(v)
    if has_inf:
        var name = "float64" if dtype == DType.float64 else "float32"
        raise Error(String("ValueError: ", subject, "contains infinity or a value too large for dtype('", name, "')."))


def _check_nonempty[dtype: DType](a: NDArray[dtype], context: String) raises:
    """Equivalent of sklearn `check_array` minimum sample/feature counts."""
    if a.ndim == 0:
        raise Error("TypeError: Input should have at least 1 dimension i.e. satisfy `len(x.shape) > 0`, got scalar instead.")
    if a.shape[0] < 1:
        raise Error(String("ValueError: Found array with 0 sample(s) (shape=", _shape_text(a),
                           ") while a minimum of 1 is required", context, "."))
    if a.ndim == 2 and a.shape[1] < 1:
        raise Error(String("ValueError: Found array with 0 feature(s) (shape=", _shape_text(a),
                           ") while a minimum of 1 is required", context, "."))


def _check_matrix[dtype: DType](a: NDArray[dtype], estimator: String, input_name: String, context: String) raises:
    """Equivalent of sklearn `check_array` defaults (ensure_2d, no nd, finite, nonempty)."""
    if a.ndim == 1:
        raise Error("ValueError: Expected 2D array, got 1D array instead. Reshape your data either using "
                    "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it "
                    "contains a single sample.")
    if a.ndim == 0:
        raise Error("ValueError: Expected 2D array, got scalar array instead. Reshape your data either using "
                    "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it "
                    "contains a single sample.")
    if a.ndim > 2:
        raise Error(String("ValueError: Found array with dim ", a.ndim, ". ", estimator, " expected <= 2."))
    _check_nonempty(a, context)
    _check_finite(a, input_name)


def _owned[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """Contiguous C-order copy with the same shape."""
    var shape = List[Int]()
    for axis in range(a.ndim):
        shape.append(a.shape[axis])
    var out = zeros[dtype](Shape(shape))
    for i in range(a.size):
        out.unsafe_set(i, _value(a, i))
    return out^


def _column[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """`a.reshape(-1, 1)` for a vector, as an owned array."""
    var out = zeros[dtype](Shape(a.shape[0], 1))
    var stride = Int(a.strides[0])
    for i in range(a.shape[0]):
        out.unsafe_set(i, a.unsafe_get(i * stride))
    return out^


def _squeeze_column[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """`a.squeeze(axis=1)` for an (n, 1) matrix, as an owned vector."""
    var out = zeros[dtype](Shape(a.shape[0]))
    var stride = Int(a.strides[0])
    for i in range(a.shape[0]):
        out.unsafe_set(i, a.unsafe_get(i * stride))
    return out^


def _every_kth_row[dtype: DType](a: NDArray[dtype], step: Int) raises -> NDArray[dtype]:
    """`a[::step]` along axis 0 (any rank), as an owned array."""
    var rows = (a.shape[0] + step - 1) // step
    var shape = List[Int]()
    shape.append(rows)
    var inner = 1
    for axis in range(1, a.ndim):
        shape.append(a.shape[axis])
        inner *= a.shape[axis]
    var out = zeros[dtype](Shape(shape))
    for r in range(rows):
        for j in range(inner):
            out.unsafe_set(r * inner + j, _value(a, r * step * inner + j))
    return out^


def _allclose[dtype: DType](a: NDArray[dtype], b: NDArray[dtype], rtol: Float64, atol: Float64) raises -> Bool:
    """`numpy.allclose(a, b, rtol, atol)` with NumPy broadcasting.

    Arithmetic runs in `dtype` (NumPy keeps Float32 against Python-float
    tolerances): |a - b| <= atol + rtol * |b| with finite b, or a == b.
    NaN is never close. Unbroadcastable shapes raise like NumPy.
    """
    var ndim = max(a.ndim, b.ndim)
    var shape = List[Int]()
    for k in range(ndim):
        var ea = a.shape[a.ndim - ndim + k] if k >= ndim - a.ndim else 1
        var eb = b.shape[b.ndim - ndim + k] if k >= ndim - b.ndim else 1
        if ea != eb and ea != 1 and eb != 1:
            raise Error(String("ValueError: operands could not be broadcast together with shapes ",
                               _shape_text(a), " ", _shape_text(b)))
        shape.append(eb if ea == 1 else ea)
    var total = 1
    for k in range(ndim):
        total *= shape[k]
    var r = Scalar[dtype](rtol)
    var t = Scalar[dtype](atol)
    for flat in range(total):
        var rest = flat
        var da = 0
        var db = 0
        for k in range(ndim - 1, -1, -1):
            var index = rest % shape[k]
            rest //= shape[k]
            var ka = k - (ndim - a.ndim)
            if ka >= 0 and a.shape[ka] != 1:
                da += index * Int(a.strides[ka])
            var kb = k - (ndim - b.ndim)
            if kb >= 0 and b.shape[kb] != 1:
                db += index * Int(b.strides[kb])
        var x = a.unsafe_get(da)
        var y = b.unsafe_get(db)
        var finite_y = not (isnan(y) or isinf(y))
        if not ((abs(x - y) <= t + r * abs(y) and finite_y) or x == y):
            return False
    return True


def _warn(mut log: List[String], message: String):
    """Emit a sklearn warning as `Category: message` on stderr and keep it."""
    print(message, file=stderr)
    log.append(message)


def _r2_score[dtype: DType](
    y_true: NDArray[dtype], y_pred: NDArray[dtype], sample_weight: Optional[NDArray[dtype]], mut log: List[String]
) raises -> Float64:
    """Equivalent of sklearn `r2_score(multioutput='uniform_average', force_finite=True)`.

    Accumulates in Float64 (sklearn computes in the input floating dtype).
    """
    if y_true.ndim > 2 or y_pred.ndim > 2:
        raise Error("ValueError: Found array with dim 3. None expected <= 2.")
    var n = y_true.shape[0]
    var lengths = List[Int]()
    lengths.append(n)
    lengths.append(y_pred.shape[0])
    if sample_weight:
        lengths.append(sample_weight.value().shape[0] if sample_weight.value().ndim > 0 else 1)
    for length in lengths:
        if length != n:
            var text = String("ValueError: Found input variables with inconsistent numbers of samples: [")
            for i in range(len(lengths)):
                text += String(lengths[i]) + (", " if i + 1 < len(lengths) else "]")
            raise Error(text)
    _check_finite(y_true, "y_true")
    _check_finite(y_pred, "y_pred")
    var outputs = y_true.shape[1] if y_true.ndim == 2 else 1
    var predicted = y_pred.shape[1] if y_pred.ndim == 2 else 1
    if outputs != predicted:
        raise Error(String("ValueError: y_true and y_pred have different number of output (",
                           outputs, "!=", predicted, ")"))
    var weights = List[Float64]()
    if sample_weight:
        var w = sample_weight.value().copy()
        if w.ndim != 1:
            raise Error("ValueError: Sample weights must be 1D array or scalar")
        _check_finite(w, "sample_weight")
        for i in range(n):
            weights.append(Float64(w.unsafe_get(i * Int(w.strides[0]))))
    else:
        for _ in range(n):
            weights.append(1.0)
    if n < 2:
        _warn(log, "UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.")
        return nan[DType.float64]()
    var mean_score = 0.0
    for k in range(outputs):
        var weight_sum = 0.0
        var weighted = 0.0
        for i in range(n):
            var flat = i * outputs + k
            weight_sum += weights[i]
            weighted += weights[i] * Float64(_value(y_true, flat))
        var average = weighted / weight_sum
        var numerator = 0.0
        var denominator = 0.0
        for i in range(n):
            var flat = i * outputs + k
            var truth = Float64(_value(y_true, flat))
            var residual = truth - Float64(_value(y_pred, flat))
            numerator += weights[i] * residual * residual
            denominator += weights[i] * (truth - average) * (truth - average)
        var score = 1.0
        if numerator != 0.0 and denominator != 0.0:
            score = 1.0 - numerator / denominator
        elif numerator != 0.0:
            score = 0.0
        mean_score += score
    return mean_score / Float64(outputs)


# ---------------------------------------------------------------------------
# Function transformer used when `transformer=None`.


@fieldwise_init
struct TargetFunctionTransformerParams[dtype: DType](Copyable, Movable):
    """Typed `get_params`/`set_params` record; field names follow sklearn."""

    var func: Optional[TargetFunction[Self.dtype]]
    var inverse_func: Optional[TargetFunction[Self.dtype]]
    var check_inverse: Bool


struct TargetFunctionTransformer[dtype: DType = DType.float64](
    Copyable, InverseTransformer, UnsupervisedEstimator
):
    """`FunctionTransformer(func, inverse_func, validate=True, check_inverse)`.

    The transformer that TransformedTargetRegressor builds from `func` and
    `inverse_func` (identity when `None`). Inputs are validated as sklearn's
    `validate=True` does: 2-D, nonempty and finite, with the fitted feature
    count enforced by `transform`. `fit` warns (UserWarning) when the round
    trip `inverse_func(func(X[::max(1, n // 100)]))` is not `allclose`
    (rtol=1e-7, atol=1e-9) to the input. Warnings go to stderr and `warnings_`.

    This is a compose-local adaptation, not `sklearn.preprocessing.FunctionTransformer`:
    it omits `accept_sparse`, `feature_names_out`, `kw_args`/`inv_kw_args` and
    `set_output`, and `validate` is fixed to True.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = TargetFunctionTransformerParams[Self.dtype]

    var func: Optional[TargetFunction[Self.dtype]]
    var inverse_func: Optional[TargetFunction[Self.dtype]]
    var check_inverse: Bool
    var n_features_in_: Optional[Int]
    var warnings_: List[String]

    def __init__(
        out self,
        func: Optional[TargetFunction[Self.dtype]] = None,
        inverse_func: Optional[TargetFunction[Self.dtype]] = None,
        *,
        check_inverse: Bool = True,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32 and Float64 targets only"
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.n_features_in_ = None
        self.warnings_ = List[String]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.func, self.inverse_func, self.check_inverse)

    def set_params(mut self, params: Self.Params) raises:
        self.func = params.func
        self.inverse_func = params.inverse_func
        self.check_inverse = params.check_inverse

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.func, self.inverse_func, check_inverse=self.check_inverse)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        _check_matrix(X, "FunctionTransformer", "X", " by FunctionTransformer")
        var log = List[String]()
        if self.check_inverse and self.func and self.inverse_func:
            var sample = _every_kth_row(X, max(1, X.shape[0] // 100))
            var round_trip = self.inverse_func.value()(self.func.value()(sample))
            if not _allclose(sample, round_trip, 1e-7, 1e-9):
                _warn(log, "UserWarning: The provided functions are not strictly inverse of each other. "
                           "If you are sure you want to proceed regardless, set 'check_inverse=False'.")
        self.n_features_in_ = X.shape[1]
        self.warnings_ = log^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _check_matrix(X, "FunctionTransformer", "X", " by FunctionTransformer")
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_.value():
            raise Error(String("ValueError: X has ", X.shape[1], " features, but FunctionTransformer is expecting ",
                               self.n_features_in_.value(), " features as input."))
        if self.func:
            return self.func.value()(X)
        return _owned(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _check_matrix(X, "Estimator", "", "")
        if self.inverse_func:
            return self.inverse_func.value()(X)
        return _owned(X)


# ---------------------------------------------------------------------------
# Component calls. Foundation associated types are only `Movable`; these
# functions state the destructibility the calls need, and callers reach them
# through `comptime if` after the NDArray type-equality checks.


def _fit_target[dtype: DType, T: TargetTransformer](mut transformer: T, y_2d: NDArray[dtype]) raises where conforms_to(
    T.Input, Deinitable
) and conforms_to(T.Target, Deinitable):
    var arg = rebind_var[T.Input](y_2d.copy())
    transformer.fit(arg)


def _apply_target[dtype: DType, T: TargetTransformer](transformer: T, a: NDArray[dtype]) raises -> NDArray[dtype] where conforms_to(
    T.Input, Deinitable
) and conforms_to(T.Transformed, Deinitable):
    var arg = rebind_var[T.Input](a.copy())
    var out = transformer.transform(arg)
    return rebind_var[NDArray[dtype]](out^)


def _invert_target[dtype: DType, T: TargetTransformer](transformer: T, a: NDArray[dtype]) raises -> NDArray[dtype] where conforms_to(
    T.Input, Deinitable
) and conforms_to(T.Transformed, Deinitable):
    var arg = rebind_var[T.Transformed](a.copy())
    var out = transformer.inverse_transform(arg)
    return rebind_var[NDArray[dtype]](out^)


def _fit_regressor[dtype: DType, R: TargetRegressor](mut regressor: R, X: R.Input, y: NDArray[dtype]) raises where conforms_to(
    R.Target, Deinitable
):
    var target = rebind_var[R.Target](y.copy())
    regressor.fit(X, target)


def _fit_regressor_weighted[dtype: DType, R: WeightedSupervisedEstimator](
    mut regressor: R, X: R.Input, y: NDArray[dtype], sample_weight: NDArray[dtype]
) raises:
    # Arguments are moved back out after the call, so their (NDArray) types
    # need no destructibility evidence here.
    var target = rebind_var[R.Target](y.copy())
    var weight = rebind_var[R.SampleWeight](sample_weight.copy())
    try:
        regressor.fit(X, target, sample_weight=weight)
    except error:
        _ = rebind_var[NDArray[dtype]](target^)
        _ = rebind_var[NDArray[dtype]](weight^)
        raise error^
    _ = rebind_var[NDArray[dtype]](target^)
    _ = rebind_var[NDArray[dtype]](weight^)


def _predict_regressor[dtype: DType, R: TargetRegressor](regressor: R, X: R.Input) raises -> NDArray[dtype] where conforms_to(
    R.Prediction, Deinitable
):
    var out = regressor.predict(X)
    return rebind_var[NDArray[dtype]](out^)


# ---------------------------------------------------------------------------
# TransformedTargetRegressor.


@fieldwise_init
struct TransformedTargetRegressorParams[dtype: DType, R: TargetRegressor, T: TargetTransformer](Copyable, Movable):
    """Typed `get_params`/`set_params` record; field names follow sklearn.

    Components are configuration copies (never fitted); their own parameters
    are reached through `regressor.value().get_params()`.
    """

    var regressor: Optional[Self.R]
    var transformer: Optional[Self.T]
    var func: Optional[TargetFunction[Self.dtype]]
    var inverse_func: Optional[TargetFunction[Self.dtype]]
    var check_inverse: Bool


struct TransformedTargetRegressor[
    dtype: DType = DType.float64,
    R: TargetRegressor = LinearRegression[dtype],
    T: TargetTransformer = TargetFunctionTransformer[dtype],
](
    Copyable,
    TransformedTargetRegressorAPI,
    WeightedSupervisedEstimator where conforms_to(R, WeightedSupervisedEstimator),
):
    """Meta-estimator to regress on a transformed target.

    `fit` computes `regressor.fit(X, transformer.transform(y))` (or `func(y)`)
    and `predict` computes `transformer.inverse_transform(regressor.predict(X))`
    (or `inverse_func(...)`). The target is promoted to a column for the
    transformer and restored to its training rank at prediction. `score` is
    R^2 in the original target space.

    Mojo adaptations (see docs/api.md):
    - `regressor=None` means LinearRegression and requires `R` to be the
      default `LinearRegression[dtype]`; `transformer=None` (identity or
      func/inverse_func) requires `T` to be `TargetFunctionTransformer[dtype]`.
      Other combinations raise `ValueError` at fit.
    - `R.Target`/`R.Prediction` and `T.Input`/`T.Transformed` must be
      `NDArray[dtype]` (checked at compile time). `X` is passed through as `R.Input`.
    - `n_features_in_` is recorded from `X.shape[1]` when `R.Input` is a dense
      matrix; named ColumnTable overloads retain `feature_names_in_`.
    - Warnings are printed to stderr and kept in `warnings_` (this fit's
      transformer warnings followed by TransformedTargetRegressor's own).
    """

    comptime Input = Self.R.Input
    comptime Target = NDArray[Self.dtype]
    comptime Prediction = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = TransformedTargetRegressorParams[Self.dtype, Self.R, Self.T]

    var regressor: Optional[Self.R]
    var transformer: Optional[Self.T]
    var func: Optional[TargetFunction[Self.dtype]]
    var inverse_func: Optional[TargetFunction[Self.dtype]]
    var check_inverse: Bool
    var regressor_: Optional[Self.R]
    var transformer_: Optional[Self.T]
    var n_features_in_: Optional[Int]
    var feature_names_in_: Optional[List[String]]
    var warnings_: List[String]
    var _training_dim: Int

    def __init__(
        out self,
        var regressor: Optional[Self.R] = None,
        *,
        var transformer: Optional[Self.T] = None,
        func: Optional[TargetFunction[Self.dtype]] = None,
        inverse_func: Optional[TargetFunction[Self.dtype]] = None,
        check_inverse: Bool = True,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32 and Float64 targets only"
        comptime assert Self.R.Target == NDArray[Self.dtype], "regressor Target must be NDArray[dtype]"
        comptime assert Self.R.Prediction == NDArray[Self.dtype], "regressor Prediction must be NDArray[dtype]"
        comptime assert Self.T.Input == NDArray[Self.dtype], "transformer Input must be NDArray[dtype]"
        comptime assert Self.T.Transformed == NDArray[Self.dtype], "transformer Transformed must be NDArray[dtype]"
        self.regressor = regressor^
        self.transformer = transformer^
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.regressor_ = None
        self.transformer_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.warnings_ = List[String]()
        self._training_dim = 0

    def __init__(out self, params: Self.Params):
        self = Self(params.regressor.copy(), transformer=params.transformer.copy(), func=params.func,
                    inverse_func=params.inverse_func, check_inverse=params.check_inverse)

    # -- parameters -------------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.regressor.copy(), self.transformer.copy(), self.func, self.inverse_func,
                           self.check_inverse)

    def set_params(mut self, params: Self.Params) raises:
        self.regressor = params.regressor.copy()
        self.transformer = params.transformer.copy()
        self.func = params.func
        self.inverse_func = params.inverse_func
        self.check_inverse = params.check_inverse

    def set_params(
        mut self,
        *,
        var regressor: Optional[Self.R] = None,
        var transformer: Optional[Self.T] = None,
        func: Optional[TargetFunction[Self.dtype]] = None,
        inverse_func: Optional[TargetFunction[Self.dtype]] = None,
        check_inverse: Optional[Bool] = None,
        regressor_params: Optional[Self.R.Params] = None,
        transformer_params: Optional[Self.T.Params] = None,
    ) raises:
        """Update only the supplied values (use the Params record to reset one to None).

        As in sklearn, components are replaced before nested
        `regressor_params`/`transformer_params` are applied to them.
        """
        if regressor_params and not regressor and not self.regressor:
            raise Error("ValueError: Invalid parameter 'regressor_params' for estimator TransformedTargetRegressor: "
                        "regressor is None.")
        if transformer_params and not transformer and not self.transformer:
            raise Error("ValueError: Invalid parameter 'transformer_params' for estimator TransformedTargetRegressor: "
                        "transformer is None.")
        if regressor:
            self.regressor = regressor^
        if transformer:
            self.transformer = transformer^
        if func:
            self.func = func
        if inverse_func:
            self.inverse_func = inverse_func
        if check_inverse:
            self.check_inverse = check_inverse.value()
        if regressor_params:
            self.regressor.value().set_params(regressor_params.value())
        if transformer_params:
            self.transformer.value().set_params(transformer_params.value())

    def __sklearn_clone__(self) raises -> Self:
        var regressor: Optional[Self.R] = None
        if self.regressor:
            regressor = clone(self.regressor.value())
        var transformer: Optional[Self.T] = None
        if self.transformer:
            transformer = clone(self.transformer.value())
        return Self(regressor^, transformer=transformer^, func=self.func, inverse_func=self.inverse_func,
                    check_inverse=self.check_inverse)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.regressor_)

    # -- fitting ----------------------------------------------------------

    def _new_transformer(self) raises -> Self.T:
        """Equivalent of sklearn `_fit_transformer` configuration checks and construction."""
        if self.transformer and (self.func or self.inverse_func):
            raise Error("ValueError: 'transformer' and functions 'func'/'inverse_func' cannot both be set.")
        if self.transformer:
            return clone(self.transformer.value())
        if Bool(self.func) != Bool(self.inverse_func):
            var lacking = "func" if not self.func else "inverse_func"
            var existing = "inverse_func" if not self.func else "func"
            raise Error(String("ValueError: When '", existing, "' is provided, '", lacking,
                               "' must also be provided. If ", lacking, " is supposed to be the default, you need "
                               "to explicitly pass it the identity function."))
        comptime if Self.T == TargetFunctionTransformer[Self.dtype]:
            return rebind_var[Self.T](TargetFunctionTransformer[Self.dtype](
                self.func, self.inverse_func, check_inverse=self.check_inverse))
        else:
            raise Error("ValueError: transformer=None selects a TargetFunctionTransformer, which this "
                        "TransformedTargetRegressor[T] cannot hold; pass a transformer or use the default T.")

    def _new_regressor(self) raises -> Self.R:
        if self.regressor:
            return clone(self.regressor.value())
        comptime if Self.R == LinearRegression[Self.dtype]:
            return rebind_var[Self.R](LinearRegression[Self.dtype]())
        else:
            raise Error("ValueError: regressor=None selects LinearRegression, which this "
                        "TransformedTargetRegressor[R] cannot hold; pass a regressor or use the default R.")

    def _transform_target(self, y: Self.Target, mut log: List[String]) raises -> Tuple[Self.T, NDArray[Self.dtype]]:
        """Validate y, fit the target transformer and return (transformer, y_trans)."""
        comptime if conforms_to(Self.T.Input, Deinitable) and conforms_to(Self.T.Target, Deinitable) and conforms_to(
            Self.T.Transformed, Deinitable
        ):
            _check_nonempty(y, "")
            _check_finite(y, "y")
            var y_2d = _column(y) if y.ndim == 1 else _owned(y)
            var transformer = self._new_transformer()
            _fit_target(transformer, y_2d)
            comptime if Self.T == TargetFunctionTransformer[Self.dtype]:
                for message in rebind[TargetFunctionTransformer[Self.dtype]](transformer).warnings_:
                    log.append(message)
            if self.check_inverse:
                var sample = _every_kth_row(y_2d, max(1, y_2d.shape[0] // 10))
                var back = _invert_target(transformer, _apply_target(transformer, sample))
                if not _allclose(sample, back, 1e-5, 1e-8):
                    _warn(log, "UserWarning: The provided functions or transformer are not strictly inverse of each "
                               "other. If you are sure you want to proceed regardless, set 'check_inverse=False'")
            var y_trans = _apply_target(transformer, y_2d)
            if y_trans.ndim == 2 and y_trans.shape[1] == 1 and y.ndim == 1:
                y_trans = _squeeze_column(y_trans)
            return (transformer^, y_trans^)
        else:
            comptime assert False, "transformer Input/Transformed must be NDArray[dtype]"

    def _commit(mut self, var transformer: Self.T, var regressor: Self.R, X: Self.Input, y: Self.Target, var log: List[String]) raises:
        var n_features: Optional[Int] = None
        comptime if Self.Input == NDArray[Self.dtype]:
            ref matrix = rebind[NDArray[Self.dtype]](X)
            if matrix.ndim == 2:
                n_features = matrix.shape[1]
        self.transformer_ = transformer^
        self.regressor_ = regressor^
        self.n_features_in_ = n_features
        self.feature_names_in_ = None
        self.warnings_ = log^
        self._training_dim = y.ndim

    def fit(mut self, X: Self.Input, y: Self.Target) raises:
        """Fit the transformer on y, then a fresh regressor clone on the transformed y.

        Learned state is replaced only after both fits succeed.
        """
        comptime if conforms_to(Self.R.Target, Deinitable):
            var log = List[String]()
            var prepared = self._transform_target(y, log)
            var regressor = self._new_regressor()
            _fit_regressor(regressor, X, prepared[1])
            self._commit(prepared[0].copy(), regressor^, X, y, log^)
        else:
            comptime assert False, "regressor Target must be NDArray[dtype]"

    def fit(
        mut self, X: Self.Input, y: Self.Target, *, sample_weight: Self.SampleWeight
    ) raises where conforms_to(Self.R, WeightedSupervisedEstimator):
        """Fit with `sample_weight` routed to the regressor (not the transformer, as in sklearn)."""
        comptime if conforms_to(Self.R.Target, Deinitable):
            var log = List[String]()
            var prepared = self._transform_target(y, log)
            var regressor = self._new_regressor()
            _fit_regressor_weighted(regressor, X, prepared[1], sample_weight)
            self._commit(prepared[0].copy(), regressor^, X, y, log^)
        else:
            comptime assert False, "regressor Target/SampleWeight must be NDArray[dtype]"

    def fit(mut self, X: ColumnTable[Self.dtype], y: Self.Target) raises where Self.Input == NDArray[Self.dtype]:
        """Fit a named numeric table, retaining names as an owned snapshot."""
        var names = X.schema().names.copy()
        self.fit(rebind[Self.Input](X.values()), y)
        self.feature_names_in_ = names^

    def fit(mut self, X: ColumnTable[Self.dtype], y: Self.Target, *, sample_weight: Self.SampleWeight) raises where Self.Input == NDArray[Self.dtype] and conforms_to(Self.R, WeightedSupervisedEstimator):
        var names = X.schema().names.copy()
        self.fit(rebind[Self.Input](X.values()), y, sample_weight=sample_weight)
        self.feature_names_in_ = names^

    def predict(self, X: ColumnTable[Self.dtype]) raises -> Self.Prediction where Self.Input == NDArray[Self.dtype]:
        check_is_fitted(self)
        if self.feature_names_in_:
            var names = X.schema().names.copy()
            if len(names) != len(self.feature_names_in_.value()):
                raise Error("ValueError: The feature names should match those that were passed during fit.")
            for i in range(len(names)):
                if names[i] != self.feature_names_in_.value()[i]:
                    raise Error("ValueError: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit.")
        return self.predict(rebind[Self.Input](X.values()))

    def score(self, X: ColumnTable[Self.dtype], y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64 where Self.Input == NDArray[Self.dtype]:
        var log = List[String]()
        return _r2_score(y, self.predict(X), sample_weight, log)

    # -- inference --------------------------------------------------------

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """`inverse_transform(regressor_.predict(X))`, restoring a 1-D training target's rank."""
        comptime if conforms_to(Self.R.Prediction, Deinitable) and conforms_to(Self.T.Input, Deinitable) and conforms_to(
            Self.T.Transformed, Deinitable
        ):
            check_is_fitted(self)
            var pred: NDArray[Self.dtype] = _predict_regressor[Self.dtype](self.regressor_.value(), X)
            if pred.ndim == 1:
                pred = _column(pred)
            var restored = _invert_target(self.transformer_.value(), pred)
            if self._training_dim == 1 and restored.ndim == 2 and restored.shape[1] == 1:
                return _squeeze_column(restored)
            return restored^
        else:
            comptime assert False, "regressor Prediction and transformer arrays must be NDArray[dtype]"

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        """R^2 of `predict(X)` against y (uniform average over outputs), as sklearn's RegressorMixin.

        An UndefinedMetricWarning for fewer than two samples is printed to stderr.
        """
        var log = List[String]()
        return _r2_score(y, self.predict(X), sample_weight, log)
