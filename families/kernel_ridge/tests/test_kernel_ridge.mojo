from moxie.kernel_ridge import KernelParameters
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from moxie.base import clone
from moxie.kernel_ridge import (
    KernelRidge,
    ScalarOrVector,
    RowKernel,
    FloatingArray,
)


@fieldwise_init
struct Metric(Copyable, RowKernel):
    var offset: Float64

    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        return X.unsafe_get(i * Int(X.strides[0])) * Y.unsafe_get(
            j * Int(Y.strides[0])
        ) + SIMD[dtype, 1](self.offset)


@fieldwise_init
struct FailingMetric(Copyable, RowKernel):
    var fail: Bool

    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        if self.fail:
            raise Error("intentional callback failure")
        return X.unsafe_get(i * Int(X.strides[0])) * Y.unsafe_get(
            j * Int(Y.strides[0])
        )


def contracts[dtype: DType]() raises:
    var X = nm.arange[dtype](3).reshape(nm.Shape(3, 1))
    var y = nm.arange[dtype](3)
    var model = KernelRidge[dtype]()
    with assert_raises():
        _ = as_float64(model.predict(X))
    _ = model.fit(X, y)
    var p = as_float64(model.predict(X))
    assert_equal(p.ndim, 1)
    assert_true(abs(p.unsafe_get(2) - Float64(5.0 / 3.0)) < 1e-5)
    assert_equal(model.n_features_in_, 1)
    var copied = model.copy()
    copied.X_fit_[nm.NDArray[dtype]].unsafe_set(0, 100)
    assert_equal(model.X_fit_[nm.NDArray[dtype]].unsafe_get(0), 0)
    var cloned = clone(model)
    assert_true(not cloned.__sklearn_is_fitted__())
    var Y = y.reshape(nm.Shape(3, 1))
    _ = model.fit(X, Y, sample_weight=2.0)
    assert_equal(as_float64(model.predict(X)).ndim, 2)
    assert_true(model.score(X, Y) > 0.95)
    var params = model.get_params_record()
    params.alpha = ScalarOrVector(0.0)
    _ = model.set_params(params)
    _ = model.fit(X, Y)
    assert_true(abs(model.score(X, Y) - 1) < 1e-5)
    var callable = KernelRidge[dtype, Metric](native_kernel=Metric(1))
    _ = callable.fit(X, y)
    assert_equal(as_float64(callable.predict(X)).shape[0], 3)
    var cc = clone(callable)
    assert_true(not cc.__sklearn_is_fitted__())
    var bad = nm.ones[dtype](nm.Shape(2))
    with assert_raises():
        _ = model.fit(X, bad)
    assert_true(model.__sklearn_is_fitted__())
    # Validation failures must not discard a previous successful fit.
    var before = as_float64(model.predict(X))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=-1.0)
    with assert_raises():
        _ = model.fit(X, y, sample_weight=ScalarOrVector(List[Float64]([1.0])))
    var nonfinite = X.copy()
    nonfinite.unsafe_set(0, SIMD[dtype, 1](Float64("nan")))
    with assert_raises():
        _ = model.fit(nonfinite, y)
    with assert_raises():
        _ = as_float64(model.predict(nm.ones[dtype](nm.Shape(2, 2))))
    with assert_raises():
        _ = model.score(X, y, ScalarOrVector(0.0))
    assert_equal(
        as_float64(model.predict(X)).unsafe_get(2), before.unsafe_get(2)
    )
    var invalid = KernelRidge[dtype](kernel="unsupported")
    with assert_raises():
        _ = invalid.fit(X, y)
    var chi = KernelRidge[dtype](kernel="chi2")
    with assert_raises():
        _ = chi.fit(X, y)
    var wrong_alpha = KernelRidge[dtype](
        ScalarOrVector(List[Float64]([1.0, 2.0]))
    )
    with assert_raises():
        _ = wrong_alpha.fit(X, y)
    var gram = KernelRidge[dtype](kernel="precomputed")
    with assert_raises():
        _ = gram.fit(X, y)
    # Snapshot ownership and native callable parameter cloning.
    var original = X.copy()
    var stable = KernelRidge[dtype]()
    _ = stable.fit(X, y)
    X.unsafe_set(0, 99)
    assert_equal(
        stable.X_fit_[nm.NDArray[dtype]].unsafe_get(0), original.unsafe_get(0)
    )
    X = original.copy()
    var cp = callable.get_params_record()
    cp.native_kernel = Metric(10)
    _ = cc.set_params(cp)
    assert_equal(callable.native_kernel.value().offset, 1)
    _ = cc.fit(X, y)
    assert_true(
        abs(
            as_float64(cc.predict(X)).unsafe_get(0)
            - as_float64(callable.predict(X)).unsafe_get(0)
        )
        > 1e-4
    )
    # A negative-stride feature view must match its logical contiguous values.
    var reverse = X.view_with_layout(
        nm.Shape(3, 1), nm.NDArrayStrides(-1, 1), 2
    )
    var contiguous = nm.zeros[dtype](nm.Shape(3, 1))
    for i in range(3):
        contiguous.unsafe_set(i, SIMD[dtype, 1](2 - i))
    _ = stable.fit(reverse, y)
    var direct = KernelRidge[dtype]()
    _ = direct.fit(contiguous, y)
    assert_true(
        abs(
            as_float64(stable.predict(reverse)).unsafe_get(0)
            - as_float64(direct.predict(contiguous)).unsafe_get(0)
        )
        < 1e-5
    )
    # Scalar score weights and vector score weights agree.
    var score1 = direct.score(contiguous, y, ScalarOrVector(2.0))
    var score2 = direct.score(
        contiguous, y, ScalarOrVector(List[Float64]([2.0, 2.0, 2.0]))
    )
    assert_true(abs(score1 - score2) < 1e-12)
    # All-zero native lists intentionally retain the numerical owner's behavior.
    var zero_model = KernelRidge[dtype]()
    _ = zero_model.fit(
        X, y, sample_weight=ScalarOrVector(List[Float64]([0.0, 0.0, 0.0]))
    )
    var zero_prediction = as_float64(zero_model.predict(X))
    for i in range(zero_prediction.size):
        assert_equal(zero_prediction.unsafe_get(i), 0)
    # Conflicting callable configuration is unavailable, not silently ignored.
    var conflict = KernelRidge[dtype, Metric](
        kernel="rbf", native_kernel=Metric(1)
    )
    with assert_raises():
        _ = conflict.fit(X, y)
    # Failed unequal-penalty factorization preserves a previous fitted model.
    var precomputed = KernelRidge[dtype](kernel="precomputed")
    var identity = nm.zeros[dtype](nm.Shape(3, 3))
    for i in range(3):
        identity.unsafe_set(i * 3 + i, 1)
    var targets = nm.ones[dtype](nm.Shape(3, 2))
    _ = precomputed.fit(identity, targets)
    var previous_coef = as_float64(precomputed.dual_coef_).copy()
    _ = precomputed.set_params(alpha=ScalarOrVector(List[Float64]([0.0, 0.1])))
    with assert_raises():
        _ = precomputed.fit(nm.zeros[dtype](nm.Shape(3, 3)), targets)
    assert_equal(
        as_float64(precomputed.dual_coef_).unsafe_get(0),
        previous_coef.unsafe_get(0),
    )
    assert_equal(precomputed.X_fit_[nm.NDArray[dtype]].unsafe_get(0), 1)
    var failing = KernelRidge[dtype, FailingMetric](
        native_kernel=FailingMetric(False)
    )
    _ = failing.fit(X, y)
    var saved = as_float64(failing.dual_coef_).copy()
    _ = failing.set_params(native_kernel=FailingMetric(True))
    with assert_raises():
        _ = failing.fit(X, y)
    assert_equal(
        as_float64(failing.dual_coef_).unsafe_get(0), saved.unsafe_get(0)
    )
    with assert_raises():
        _ = as_float64(failing.predict(X))
    var nonfinite_metric = KernelRidge[dtype, Metric](
        native_kernel=Metric(Float64("inf"))
    )
    with assert_raises():
        _ = nonfinite_metric.fit(X, y)
    var disabled = failing.get_params_record()
    disabled.native_kernel = None
    _ = failing.set_params(disabled)
    _ = failing.fit(X, y)
    print("contracts", dtype)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    prediction_params[DType.float32]()
    prediction_params[DType.float64]()
    singleton_score_weights[DType.float32]()
    singleton_score_weights[DType.float64]()


def prediction_params[dtype: DType]() raises:
    var X = nm.arange[dtype](3).reshape(nm.Shape(3, 1))
    var y = nm.arange[dtype](3)
    var model = KernelRidge[dtype]()
    _ = model.fit(X, y)
    var expected = as_float64(model.predict(X))
    var original = model.get_params_record()
    for parameter in range(3):
        _ = model.set_params(original)
        if parameter == 0:
            _ = model.set_params(gamma=-1.0)
        elif parameter == 1:
            _ = model.set_params(degree=-1.0)
        else:
            _ = model.set_params(coef0=Float64("inf"))
        var actual = as_float64(model.predict(X))
        for i in range(actual.size):
            assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))
        var cloned = clone(model)
        assert_true(not cloned.__sklearn_is_fitted__())
        with assert_raises():
            _ = cloned.fit(X, y)
        with assert_raises():
            _ = model.fit(X, y)
        assert_equal(
            as_float64(model.predict(X)).unsafe_get(2), expected.unsafe_get(2)
        )
    _ = model.set_params(original)
    _ = model.set_params(kernel="rbf", gamma=-1.0)
    with assert_raises():
        _ = as_float64(model.predict(X))
    _ = model.set_params(kernel="unsupported")
    with assert_raises():
        _ = as_float64(model.predict(X))
    # Named updates replace the active callable; explicit conflicts still fail.
    var callable = KernelRidge[dtype, Metric](native_kernel=Metric(1))
    _ = callable.fit(X, y)
    _ = callable.set_params(kernel="rbf")
    assert_true(not callable.native_kernel)
    _ = as_float64(callable.predict(X))
    _ = callable.set_params(kernel=Metric(1))
    var extras: KernelParameters = {"unused": 1.0}
    _ = callable.set_params(kernel_params=extras^)
    with assert_raises():
        _ = as_float64(callable.predict(X))


def singleton_score_weights[dtype: DType]() raises:
    var X = nm.ones[dtype](nm.Shape(1, 1))
    var y = nm.ones[dtype](nm.Shape(1))
    var model = KernelRidge[dtype]()
    _ = model.fit(X, y)
    var before = as_float64(model.dual_coef_).unsafe_get(0)
    assert_true(isnan(model.score(X, y)))
    assert_true(isnan(model.score(X, y, ScalarOrVector(2.0))))
    assert_true(isnan(model.score(X, y, ScalarOrVector(-1.0))))
    with assert_raises():
        _ = model.score(X, y, ScalarOrVector(0.0))
    with assert_raises():
        _ = model.score(X, y, ScalarOrVector(List[Float64]([0.0])))
    with assert_raises():
        _ = model.score(X, y, ScalarOrVector(Float64("nan")))
    with assert_raises():
        _ = model.score(X, y, ScalarOrVector(List[Float64]([1.0, 1.0])))
    assert_equal(as_float64(model.dual_coef_).unsafe_get(0), before)


def as_float64(value: FloatingArray) raises -> nm.NDArray[DType.float64]:
    if value.isa[nm.NDArray[DType.float32]]():
        return value[nm.NDArray[DType.float32]].astype[DType.float64]()
    return value[nm.NDArray[DType.float64]].copy()
