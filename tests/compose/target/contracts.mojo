"""Native contracts for TransformedTargetRegressor and TargetFunctionTransformer."""
from numojo import NDArray, Shape, zeros
from std.math import log, exp, isnan, nan, inf
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal, assert_raises, TestSuite
from moxie.base import BaseEstimator, InverseTransformer, UnsupervisedEstimator, clone
from moxie.utils.validation import check_is_fitted
from moxie.linear_model import LinearRegression, LinearRegressionParams
from moxie.compose import (
    TransformedTargetRegressor,
    TransformedTargetRegressorParams,
    TargetFunctionTransformer,
    TargetFunctionTransformerParams,
)

comptime F64 = DType.float64
comptime F32 = DType.float32


# -- fixtures -----------------------------------------------------------------


def matrix[dtype: DType = F64](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var a = zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def vector[dtype: DType = F64](values: List[Float64]) raises -> NDArray[dtype]:
    var a = zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def features() raises -> NDArray[F64]:
    return matrix(6, 2, [0.0, 1.0, 1.0, 0.5, 2.0, 2.0, 3.0, 1.5, 4.0, 3.0, 5.0, 2.5])


def positive_target() raises -> NDArray[F64]:
    return vector([1.0, 2.0, 4.5, 6.0, 11.0, 14.0])


def apply[dtype: DType](x: NDArray[dtype], op: Int) raises -> NDArray[dtype]:
    var shape = List[Int]()
    for axis in range(x.ndim):
        shape.append(x.shape[axis])
    var out = zeros[dtype](Shape(shape))
    for i in range(x.size):
        var v = Float64(x.unsafe_get(i))
        var r: Float64
        if op == 0:
            r = log(v)
        elif op == 1:
            r = exp(v)
        elif op == 2:
            r = v + 1.0
        else:
            r = v
        out.unsafe_set(i, r.cast[dtype]())
    return out^


def log_f(x: NDArray[F64]) raises -> NDArray[F64]:
    return apply(x, 0)


def exp_f(x: NDArray[F64]) raises -> NDArray[F64]:
    return apply(x, 1)


def plus_one(x: NDArray[F64]) raises -> NDArray[F64]:
    return apply(x, 2)


def same(x: NDArray[F64]) raises -> NDArray[F64]:
    return apply(x, 3)


def log_f32(x: NDArray[F32]) raises -> NDArray[F32]:
    return apply(x, 0)


def exp_f32(x: NDArray[F32]) raises -> NDArray[F32]:
    return apply(x, 1)


def squeeze_f(x: NDArray[F64]) raises -> NDArray[F64]:
    """Returns a vector (sklearn tolerates 1-D func output during fit)."""
    var out = zeros[F64](Shape(x.shape[0]))
    for i in range(x.shape[0]):
        out.unsafe_set(i, x.unsafe_get(i * Int(x.strides[0])))
    return out^


@fieldwise_init
struct ScaleParams(Copyable, Movable):
    var factor: Float64


struct Scale(Copyable, InverseTransformer, UnsupervisedEstimator):
    """Test double: multiplies by a constant; records how often it was fitted."""

    comptime Input = NDArray[F64]
    comptime Target = NDArray[F64]
    comptime Transformed = NDArray[F64]
    comptime Params = ScaleParams
    var factor: Float64
    var fits_: Int

    def __init__(out self, factor: Float64 = 2.0):
        self.factor = factor
        self.fits_ = 0

    def get_params(self, deep: Bool = True) raises -> ScaleParams:
        return ScaleParams(self.factor)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: ScaleParams,
    ) raises -> ref[self] Self:
        self.factor = params.factor
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.factor)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fits_ > 0

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[F64],
        y: Optional[NDArray[F64]] = None,
    ) raises -> ref[self] Self:
        if X.ndim != 2:
            raise Error("ValueError: Scale expects a matrix")
        self.fits_ += 1
        return self

    def transform(self, X: NDArray[F64]) raises -> NDArray[F64]:
        var out = X.copy()
        for i in range(out.size):
            out.unsafe_set(i, out.unsafe_get(i) * self.factor)
        return out^

    def inverse_transform(self, X: NDArray[F64]) raises -> NDArray[F64]:
        var out = X.copy()
        for i in range(out.size):
            out.unsafe_set(i, out.unsafe_get(i) / self.factor)
        return out^


def assert_close[dtype: DType](a: NDArray[dtype], b: NDArray[dtype], atol: Float64 = 1e-10) raises:
    assert_equal(a.ndim, b.ndim)
    for axis in range(a.ndim):
        assert_equal(a.shape[axis], b.shape[axis])
    for i in range(a.size):
        assert_almost_equal(Float64(a.unsafe_get(i)), Float64(b.unsafe_get(i)), atol=atol)


def expect_error(message: String, err: Error) raises:
    assert_true(String(err).startswith(message), String("expected '", message, "', got '", err, "'"))


# -- construction and parameters ---------------------------------------------


def test_defaults_and_params() raises:
    var model = TransformedTargetRegressor()
    var params = model.get_params()
    assert_false(Bool(params.regressor))
    assert_false(Bool(params.transformer))
    assert_false(Bool(params.func))
    assert_false(Bool(params.inverse_func))
    assert_true(params.check_inverse)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(features())


def test_identity_matches_linear_regression() raises:
    var X = features()
    var y = positive_target()
    var model = TransformedTargetRegressor()
    _ = model.fit(X, y)
    var reference = LinearRegression()
    _ = reference.fit(X, y)
    var pred = model.predict(X)
    assert_equal(pred.ndim, 1)
    assert_close(pred, reference.predict(X))
    assert_equal(model.n_features_in_.value(), 2)
    assert_equal(len(model.warnings_), 0)
    # The default transformer_ is an identity TargetFunctionTransformer.
    assert_true(model.transformer_.value().__sklearn_is_fitted__())
    assert_false(Bool(model.transformer_.value().func))
    assert_close(model.regressor_.value().coef_, reference.coef_)


def test_function_pair() raises:
    var X = features()
    var y = positive_target()
    var model = TransformedTargetRegressor(func=log_f, inverse_func=exp_f)
    _ = model.fit(X, y)
    var reference = LinearRegression()
    _ = reference.fit(X, log_f(y))
    assert_close(model.regressor_.value().coef_, reference.coef_)
    assert_close(model.predict(X), exp_f(reference.predict(X)))
    assert_equal(len(model.warnings_), 0)
    var r2 = model.score(X, y)
    assert_true(r2 > 0.9 and r2 <= 1.0)


def test_target_shapes() raises:
    var X = features()
    var column = matrix(6, 1, [1.0, 2.0, 4.5, 6.0, 11.0, 14.0])
    var model = TransformedTargetRegressor(func=log_f, inverse_func=exp_f)
    _ = model.fit(X, column)
    var pred = model.predict(X)
    assert_equal(pred.ndim, 2)
    assert_equal(pred.shape[1], 1)
    var two = matrix(6, 2, [1.0, 3.0, 2.0, 1.0, 4.5, 2.0, 6.0, 5.0, 11.0, 4.0, 14.0, 8.0])
    _ = model.fit(X, two)
    pred = model.predict(X)
    assert_equal(pred.shape[0], 6)
    assert_equal(pred.shape[1], 2)
    var reference = LinearRegression()
    _ = reference.fit(X, log_f(two))
    assert_close(pred, exp_f(reference.predict(X)))


def test_vector_function_output() raises:
    # func may return a vector; regression proceeds on it (sklearn FIXME path).
    var model = TransformedTargetRegressor(func=squeeze_f, inverse_func=same, check_inverse=False)
    _ = model.fit(features(), positive_target())
    assert_equal(model.predict(features()).ndim, 1)
    # With check_inverse, the 1-D forward output reaches inverse_transform's 2-D check.
    var checked = TransformedTargetRegressor(func=squeeze_f, inverse_func=same)
    try:
        _ = checked.fit(features(), positive_target())
        raise Error("expected failure")
    except err:
        expect_error("ValueError: Expected 2D array, got 1D array instead", err)


def test_inverse_warnings() raises:
    var X = features()
    var y = positive_target()
    var model = TransformedTargetRegressor(func=plus_one, inverse_func=same)
    _ = model.fit(X, y)
    assert_equal(len(model.warnings_), 2)
    assert_true(model.warnings_[0].startswith("UserWarning: The provided functions are not strictly inverse"))
    assert_true(model.warnings_[1].startswith("UserWarning: The provided functions or transformer are not strictly"))
    assert_true(model.transformer_.value().warnings_[0].startswith("UserWarning: The provided functions are"))
    var quiet = TransformedTargetRegressor(func=plus_one, inverse_func=same, check_inverse=False)
    _ = quiet.fit(X, y)
    assert_equal(len(quiet.warnings_), 0)
    # A non-inverse custom transformer warns only through the meta-estimator check.
    var scaled = TransformedTargetRegressor[F64, LinearRegression[F64], Scale](transformer=Scale(2.0))
    _ = scaled.fit(X, y)
    assert_equal(len(scaled.warnings_), 0)


def test_configuration_errors() raises:
    var X = features()
    var y = positive_target()
    var both = TransformedTargetRegressor[F64, LinearRegression[F64], Scale](transformer=Scale(), func=log_f, inverse_func=exp_f)
    try:
        _ = both.fit(X, y)
        raise Error("expected failure")
    except err:
        expect_error("ValueError: 'transformer' and functions 'func'/'inverse_func' cannot both be set.", err)
    var forward_only = TransformedTargetRegressor(func=log_f)
    try:
        _ = forward_only.fit(X, y)
        raise Error("expected failure")
    except err:
        expect_error("ValueError: When 'func' is provided, 'inverse_func' must also be provided.", err)
    var inverse_only = TransformedTargetRegressor(inverse_func=exp_f)
    try:
        _ = inverse_only.fit(X, y)
        raise Error("expected failure")
    except err:
        expect_error("ValueError: When 'inverse_func' is provided, 'func' must also be provided.", err)
    # Static component types cannot hold the implicit defaults.
    var custom_t = TransformedTargetRegressor[F64, LinearRegression[F64], Scale]()
    with assert_raises(contains="transformer=None selects a TargetFunctionTransformer"):
        _ = custom_t.fit(X, y)
    assert_false(custom_t.__sklearn_is_fitted__())


def test_target_validation() raises:
    var X = features()
    var model = TransformedTargetRegressor()
    var y = positive_target()
    y.unsafe_set(2, nan[F64]())
    with assert_raises(contains="ValueError: Input y contains NaN."):
        _ = model.fit(X, y)
    y.unsafe_set(2, inf[F64]())
    with assert_raises(contains="ValueError: Input y contains infinity or a value too large for dtype('float64')."):
        _ = model.fit(X, y)
    with assert_raises(contains="ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."):
        _ = model.fit(X, zeros[F64](Shape(0)))
    with assert_raises(contains="ValueError: Found array with 0 feature(s) (shape=(6, 0))"):
        _ = model.fit(X, zeros[F64](Shape(6, 0)))
    # Function transformers reject targets of rank three.
    with assert_raises(contains="ValueError: Found array with dim 3. FunctionTransformer expected <= 2."):
        _ = model.fit(X, zeros[F64](Shape(6, 1, 1)))
    # Regressor errors propagate (sample-count mismatch).
    with assert_raises():
        _ = model.fit(X, vector([1.0, 2.0, 3.0]))
    assert_false(model.__sklearn_is_fitted__())


def test_repeat_and_failed_fit() raises:
    var X = features()
    var model = TransformedTargetRegressor(func=log_f, inverse_func=exp_f)
    _ = model.fit(X, positive_target())
    var first = model.predict(X)
    var other = vector([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    _ = model.fit(X, other)
    var second = model.predict(X)
    assert_true(abs(Float64(second.unsafe_get(0)) - Float64(first.unsafe_get(0))) > 1e-3)
    # A failing fit keeps the previous learned state.
    var bad = other.copy()
    bad.unsafe_set(0, -1.0)  # log(-1) = NaN reaches the regressor
    var failed = False
    try:
        _ = model.fit(X, bad)
    except:
        failed = True
    assert_true(failed, "NaN transformed targets must fail the regressor fit")
    assert_close(model.predict(X), second)
    _ = model.set_params(func=plus_one, inverse_func=same)
    _ = model.fit(X, other)
    assert_equal(len(model.warnings_), 2)
    _ = model.set_params(check_inverse=False)
    _ = model.fit(X, other)
    assert_equal(len(model.warnings_), 0)


def test_clone_and_params() raises:
    var X = features()
    var y = positive_target()
    var model = TransformedTargetRegressor(LinearRegression(fit_intercept=False), func=log_f, inverse_func=exp_f)
    _ = model.fit(X, y)
    # The constructor regressor is configuration: fitting used a clone.
    assert_false(model.regressor.value().__sklearn_is_fitted__())
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_false(copy.get_params().regressor.value().fit_intercept)
    _ = copy.set_params(regressor_params=LinearRegressionParams(True, True, 1e-6, None, False))
    assert_true(copy.regressor.value().fit_intercept)
    assert_false(model.regressor.value().fit_intercept)
    assert_true(model.__sklearn_is_fitted__())
    # Whole-component replacement precedes nested parameter updates.
    _ = copy.set_params(regressor=LinearRegression(tol=1e-3), regressor_params=LinearRegressionParams(False, True, 1e-4, None, False))
    assert_equal(copy.regressor.value().tol, 1e-4)
    # Records restore None components and round-trip all fields.
    var record = model.get_params()
    record.regressor = None
    _ = copy.set_params(record)
    assert_false(Bool(copy.regressor))
    assert_true(Bool(copy.func))
    var empty = TransformedTargetRegressor()
    with assert_raises(contains="Invalid parameter 'regressor_params'"):
        _ = empty.set_params(regressor_params=LinearRegressionParams(True, True, 1e-6, None, False))
    var rebuilt = TransformedTargetRegressor(model.get_params())
    _ = rebuilt.fit(X, y)
    assert_close(rebuilt.predict(X), model.predict(X))


def test_custom_transformer() raises:
    var X = features()
    var y = positive_target()
    var model = TransformedTargetRegressor[F64, LinearRegression[F64], Scale](transformer=Scale(4.0))
    _ = model.fit(X, y)
    assert_equal(model.transformer_.value().fits_, 1)
    assert_equal(model.transformer.value().fits_, 0)
    var reference = LinearRegression()
    _ = reference.fit(X, Scale(4.0).transform(matrix(6, 1, [1.0, 2.0, 4.5, 6.0, 11.0, 14.0])))
    assert_close(model.predict(X), squeeze_f(Scale(4.0).inverse_transform(reference.predict(X))))
    _ = model.set_params(transformer_params=ScaleParams(8.0))
    assert_equal(model.transformer.value().factor, 8.0)


def test_sample_weight_and_score() raises:
    var X = features()
    var y = positive_target()
    var w = vector([1.0, 0.0, 2.0, 1.0, 3.0, 0.5])
    var model = TransformedTargetRegressor(func=log_f, inverse_func=exp_f)
    _ = model.fit(X, y, sample_weight=w)
    var reference = LinearRegression()
    _ = reference.fit(X, log_f(y), sample_weight=w)
    var pred = exp_f(reference.predict(X))
    assert_close(model.predict(X), pred)
    # R^2 (weighted and unweighted) in the original target space.
    var mean = 0.0
    var wsum = 0.0
    for i in range(6):
        mean += Float64(w.unsafe_get(i)) * Float64(y.unsafe_get(i))
        wsum += Float64(w.unsafe_get(i))
    mean /= wsum
    var num = 0.0
    var den = 0.0
    for i in range(6):
        var r = Float64(y.unsafe_get(i)) - Float64(pred.unsafe_get(i))
        num += Float64(w.unsafe_get(i)) * r * r
        den += Float64(w.unsafe_get(i)) * (Float64(y.unsafe_get(i)) - mean) ** 2
    assert_almost_equal(model.score(X, y, w.copy()), 1.0 - num / den, atol=1e-12)
    with assert_raises(contains="inconsistent numbers of samples: [6, 6, 2]"):
        _ = model.score(X, y, Optional(vector([1.0, 2.0])))
    var single = model.score(matrix(1, 2, [1.0, 1.0]), vector([3.0]))
    assert_true(isnan(single))


def test_float32() raises:
    var X = matrix[F32](6, 2, [0.0, 1.0, 1.0, 0.5, 2.0, 2.0, 3.0, 1.5, 4.0, 3.0, 5.0, 2.5])
    var y = vector[F32]([1.0, 2.0, 4.5, 6.0, 11.0, 14.0])
    var model = TransformedTargetRegressor[F32](func=log_f32, inverse_func=exp_f32)
    _ = model.fit(X, y)
    var reference = LinearRegression[F32]()
    _ = reference.fit(X, log_f32(y))
    assert_close(model.predict(X), exp_f32(reference.predict(X)), atol=1e-4)
    y.unsafe_set(0, inf[F32]())
    with assert_raises(contains="dtype('float32')"):
        _ = model.fit(X, y)


def test_function_transformer_alone() raises:
    var ft = TargetFunctionTransformer(log_f, exp_f)
    with assert_raises(contains="NotFittedError"):
        _ = ft.transform(matrix(2, 1, [1.0, 2.0]))
    _ = ft.fit(matrix(3, 1, [1.0, 2.0, 3.0]))
    assert_equal(ft.n_features_in_.value(), 1)
    with assert_raises(contains="X has 2 features, but FunctionTransformer is expecting 1 features as input."):
        _ = ft.transform(matrix(1, 2, [1.0, 2.0]))
    with assert_raises(contains="Expected 2D array, got 1D array instead"):
        _ = ft.fit(vector([1.0, 2.0]))
    var params = ft.get_params()
    assert_true(params.check_inverse)
    var identity = clone(ft)
    _ = identity.set_params(TargetFunctionTransformerParams[F64](None, None, False))
    _ = identity.fit(matrix(2, 1, [5.0, 7.0]))
    assert_close(identity.transform(matrix(2, 1, [5.0, 7.0])), matrix(2, 1, [5.0, 7.0]))
    assert_close(identity.inverse_transform(matrix(2, 1, [5.0, 7.0])), matrix(2, 1, [5.0, 7.0]))
    with assert_raises(contains="Input contains NaN."):
        _ = identity.inverse_transform(matrix(1, 1, [nan[F64]()]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
