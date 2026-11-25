from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from moxie.ensemble._hist_gradient_boosting.losses import _ClassificationLoss
from numojo import NDArray, NDArrayStrides, Shape

comptime Values = NDArray[DType.float64]


def test_class_major_buffers_and_strides() raises:
    var loss = _ClassificationLoss(3)
    var y = Values(Shape(2))
    y.unsafe_set(0, 0)
    y.unsafe_set(1, 2)
    var base = Values(Shape(2, 3))
    for i in range(base.size):
        base.unsafe_set(i, 0)
    var raw = base.view_with_layout(Shape(2, 3), NDArrayStrides(-3, -1), 5)
    var gradient = List[Float32]()
    var hessian = List[Float32]()
    loss.gradient_hessian(y, raw, None, gradient, hessian)
    comptime low = Float32(-2.0 / 3.0)
    comptime high = Float32(1.0 / 3.0)
    assert_equal(gradient, [low, high, high, high, high, low])
    assert_equal(hessian, List[Float32](length=6, fill=Float32(2.0 / 9.0)))
    raw.unsafe_set(0, 1000)
    assert_equal(loss.predict_proba(raw).item(0, 0), 1.0)
    assert_equal(loss.predict_proba(raw).item(1, 2), 1.0 / 3.0)
    var single_raw = Values(Shape(2, 1))
    single_raw.unsafe_set(0, 50)
    single_raw.unsafe_set(1, -50)
    y.unsafe_set(1, 1)
    _ClassificationLoss(2).gradient_hessian(y, single_raw, None, gradient, hessian)
    assert_equal(gradient, [Float32(1), Float32(-1)])
    assert_equal(len(hessian), 2)
    assert_true(hessian[0] > 0 and hessian[1] > 0)


def test_validation_and_single_class() raises:
    with assert_raises(contains="at least one class"):
        _ = _ClassificationLoss(0)
    var loss = _ClassificationLoss(1)
    var y = Values(Shape(1))
    y.unsafe_set(0, 0)
    assert_true(loss.fit_intercept(y).item(0) < -30)
    var weights = Values(Shape(1))
    weights.unsafe_set(0, 0)
    with assert_raises(contains="sum to zero"):
        _ = loss.fit_intercept(y, weights.copy())
    weights.unsafe_set(0, -1)
    with assert_raises(contains="nonnegative"):
        _ = loss.fit_intercept(y, weights^)
    y.unsafe_set(0, 0.5)
    with assert_raises(contains="encoded class index"):
        _ = loss.fit_intercept(y)
    with assert_raises(contains="one column"):
        _ = loss.predict_proba(y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
