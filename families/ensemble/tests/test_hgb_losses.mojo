from std.math import isnan, nan
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from moxie.ensemble._hist_gradient_boosting.losses import _RegressionLoss
from numojo import NDArray, NDArrayStrides, Shape


def vector(values: List[Float64]) raises -> NDArray[DType.float64]:
    var result = NDArray[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def test_weighted_and_unweighted_location() raises:
    var loss = _RegressionLoss("absolute_error")
    var y = vector([40.0, 4.0, 1.0, 10.0])
    var w = vector([0.0, 1.0, 1.0, 0.0])
    assert_equal(loss.fit_intercept(y), 7.0)
    assert_equal(loss.fit_intercept(y, w^), 1.0)
    assert_true(isnan(loss.fit_intercept(y, vector([0.0, 0.0, 0.0, 0.0]))))
    assert_equal(_RegressionLoss("quantile", 0.25).fit_intercept(y), 3.25)


def test_strided_vectors_and_reused_derivative_buffers() raises:
    var loss = _RegressionLoss("squared_error")
    var base = vector([8.0, 100.0, 4.0, 100.0, 0.0])
    var y = base.view_with_layout(Shape(3), NDArrayStrides(-2), 4)
    var raw = vector([1.0, 3.0, 7.0])
    var weights = vector([0.5, 1.5, 2.0])
    var w: Optional[NDArray[DType.float64]] = weights.view_with_layout(
        Shape(3), NDArrayStrides(-1), 2
    )
    assert_equal(loss.fit_intercept(y), 4.0)
    assert_equal(loss.fit_intercept(y, w), 2.5)
    assert_equal(loss.mean_loss(y, raw, w), 0.5)
    var gradient = List[Float32]()
    var hessian = List[Float32]()
    loss.gradient_hessian(y, raw, w, gradient, hessian)
    assert_equal(gradient, [Float32(2), Float32(-1.5), Float32(-0.5)])
    assert_equal(hessian, [Float32(2), Float32(1.5), Float32(0.5)])
    loss.gradient_hessian(y, raw, None, gradient, hessian)
    assert_equal(gradient, [Float32(1), Float32(-1), Float32(-1)])
    assert_equal(len(hessian), 0)


def test_target_and_buffer_validation() raises:
    with assert_raises(contains="unknown"):
        _ = _RegressionLoss("typo")
    with assert_raises(contains="quantile"):
        _ = _RegressionLoss("quantile")
    with assert_raises(contains="quantile"):
        _ = _RegressionLoss("quantile", 1.0)
    with assert_raises(contains="positive target sum"):
        _ = _RegressionLoss("poisson").fit_intercept(vector([0.0, 0.0]))
    with assert_raises(contains="nonnegative targets"):
        _ = _RegressionLoss("poisson").fit_intercept(vector([1.0, -1.0]))
    with assert_raises(contains="strictly positive"):
        _ = _RegressionLoss("gamma").fit_intercept(vector([0.0, 2.0]))
    var loss = _RegressionLoss("squared_error")
    with assert_raises(contains="finite"):
        _ = loss.fit_intercept(vector([nan[DType.float64]()]))
    with assert_raises(contains="one entry"):
        _ = loss.mean_loss(vector([1.0]), vector([1.0]), vector([1.0, 2.0]))
    with assert_raises(contains="nonnegative"):
        _ = loss.fit_intercept(vector([1.0]), vector([-1.0]))
    with assert_raises(contains="sum to zero"):
        _ = loss.fit_intercept(vector([1.0]), vector([0.0]))
    var gradients: List[Float32] = [43.0]
    var hessians: List[Float32] = [6.0]
    with assert_raises(contains="matching"):
        loss.gradient_hessian(vector([1.0]), vector([1.0, 2.0]), None, gradients, hessians)
    assert_equal(gradients, [Float32(43)])
    assert_equal(hessians, [Float32(6)])


def test_log_link_preserves_nan_mean() raises:
    var loss = _RegressionLoss("gamma")
    var y = vector([1e308, 1e308])
    var w = vector([1e308, 1e308])
    # Both the weighted sum and the sum of weights overflow.
    assert_true(isnan(loss.fit_intercept(y, w^)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
