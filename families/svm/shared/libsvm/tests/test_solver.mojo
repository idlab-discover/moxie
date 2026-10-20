"""Behavioral engine and cache contracts, using the actual native solver."""
import numojo as nm
from std.math import abs
from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from moxie.svm._contracts import KernelSpec, KernelOptions
from moxie.svm._cache import ColumnCache
from moxie.svm._kernels import prediction_kernel_matrix
from moxie.svm._libsvm import LibSVM


def test_linear_classification_and_regression() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    x.unsafe_set(0, -2)
    x.unsafe_set(1, -1)
    x.unsafe_set(2, 1)
    x.unsafe_set(3, 2)
    var y = nm.zeros[DType.float64](nm.Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    var weights = nm.ones[DType.float64](nm.Shape(4))
    var classes = nm.ones[DType.float64](nm.Shape(2))
    var options = KernelOptions("c_svc", KernelSpec("linear", 1, 3, 0), 1, 0.5, 0.1, 1e-6, 0.000001, True, False, -1, 17)
    var engine = LibSVM()
    var model = engine.solve(x, y, weights, classes, options)
    assert_equal(model.support.size, 2)
    assert_equal(model.support.unsafe_get(0), 1)
    assert_equal(model.support.unsafe_get(1), 2)
    assert_true(abs(model.dual_coef.unsafe_get(0) - 0.5) < 1e-12)
    assert_true(abs(model.dual_coef.unsafe_get(1) + 0.5) < 1e-12)
    var decision = engine.decision_values(x, model, options.kernel)
    for i in range(4):
        assert_true(abs(decision.unsafe_get(i) + x.unsafe_get(i)) < 1e-12)
    options.formulation = "epsilon_svr"
    for i in range(4):
        y.unsafe_set(i, x.unsafe_get(i))
    var empty = nm.zeros[DType.float64](nm.Shape(0))
    var regression = engine.solve(x, y, weights, empty, options)
    var predictions = engine.decision_values(x, regression, options.kernel)
    for i in range(4):
        assert_true(abs(predictions.unsafe_get(i) - 0.95 * x.unsafe_get(i)) < 1e-12)


def test_cache_eviction_and_owned_columns() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    for i in range(4):
        x.unsafe_set(i, Float64(i + 1))
    var cache = ColumnCache(x, KernelSpec("linear", 1, 3, 0), 0.000001)
    var first = cache.column(0)
    var second = cache.column(1)
    var third = cache.column(2)
    assert_equal(cache.capacity, 2)
    assert_equal(len(cache.columns), 2)
    assert_equal(first.unsafe_get(3), Float32(4))
    assert_equal(second.unsafe_get(3), Float32(8))
    assert_equal(third.unsafe_get(3), Float32(12))
    var again = cache.column(0)
    assert_equal(again.unsafe_get(3), Float32(4))
    assert_equal(len(cache.columns), 2)
    var huge = ColumnCache(x, KernelSpec("linear", 1, 3, 0), 1e300)
    assert_equal(huge.capacity, 4)


def test_nu_and_one_class_constraints() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 1))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        x.unsafe_set(i, Float64(i) - 2.5)
        y.unsafe_set(i, Float64(i >= 3))
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var classes = nm.ones[DType.float64](nm.Shape(2))
    var options = KernelOptions("nu_svc", KernelSpec("rbf", 0.5, 3, 0), 1, 0.5, 0.1, 1e-5, 0.00001, True, False, -1, 17)
    var engine = LibSVM()
    var model = engine.solve(x, y, weights, classes, options)
    var decision = engine.decision_values(x, model, options.kernel)
    for i in range(6):
        assert_true((decision.unsafe_get(i) > 0) == (i < 3))
    options.formulation = "one_class"
    var empty = nm.zeros[DType.float64](nm.Shape(0))
    var outlier = engine.solve(x, y, weights, empty, options)
    var mass = Float64(0)
    for i in range(outlier.dual_coef.size):
        mass += outlier.dual_coef.unsafe_get(i)
    assert_true(abs(mass - 3) < 1e-12)
    options.formulation = "nu_svr"
    var regression = engine.solve(x, y, weights, empty, options)
    mass = 0
    for i in range(regression.dual_coef.size):
        mass += regression.dual_coef.unsafe_get(i)
    assert_true(abs(mass) < 1e-12)


def test_nonpositive_weights_and_iteration_limit() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    var weights = nm.ones[DType.float64](nm.Shape(4))
    for i in range(4):
        x.unsafe_set(i, Float64(i) - 1.5)
        y.unsafe_set(i, Float64(i >= 2))
    weights.unsafe_set(0, 0)
    var classes = nm.ones[DType.float64](nm.Shape(2))
    var options = KernelOptions("c_svc", KernelSpec("linear", 1, 3, 0), 1, 0.5, 0.1, 1e-6, 1, True, False, 0, 17)
    var engine = LibSVM()
    var limited = engine.solve(x, y, weights, classes, options)
    assert_equal(limited.fit_status, 1)
    assert_equal(limited.iterations.unsafe_get(0), 0)
    options.max_iter = -1
    var fitted = engine.solve(x, y, weights, classes, options)
    for i in range(fitted.support.size):
        assert_true(fitted.support.unsafe_get(i) >= 1)
    weights.unsafe_set(1, -1)
    with assert_raises():
        _ = engine.solve(x, y, weights, classes, options)


def test_rbf_prediction_avoids_norm_cancellation() raises:
    var x = nm.zeros[DType.float64](nm.Shape(1, 1))
    var train = nm.zeros[DType.float64](nm.Shape(1, 1))
    x.unsafe_set(0, 1000000001)
    train.unsafe_set(0, 1000000000)
    var matrix = prediction_kernel_matrix(x, train, KernelSpec("rbf", 0.5, 3, 0))
    assert_true(abs(matrix.unsafe_get(0) - Float64(0.6065306597126334)) < 1e-15)


def test_constant_rbf_decision_preserves_reference_sign() raises:
    var x = nm.full[DType.float64](nm.Shape(12, 3), 0.5)
    var y = nm.zeros[DType.float64](nm.Shape(12))
    for i in range(12):
        y.unsafe_set(i, Float64(i % 2))
    var weights = nm.ones[DType.float64](nm.Shape(12))
    var classes = nm.ones[DType.float64](nm.Shape(2))
    var options = KernelOptions("c_svc", KernelSpec("rbf", 1, 3, 0), 1, 0.5, 0.1, 1e-6, 1, True, False, -1, 17)
    var query = nm.zeros[DType.float64](nm.Shape(2, 3))
    query.unsafe_set(0, -0.062457128643400986)
    query.unsafe_set(1, 0.020783000089848524)
    query.unsafe_set(2, 0.034668141095009423)
    query.unsafe_set(3, -0.08905790646767567)
    query.unsafe_set(4, 0.4031274031714913)
    query.unsafe_set(5, 0.39573236780040666)
    var engine = LibSVM()
    var model = engine.solve(x, y, weights, classes, options)
    var decisions = engine.decision_values(query, model, options.kernel)
    assert_equal(decisions.unsafe_get(0), Float64(-1.1102230246251565e-16))
    assert_equal(decisions.unsafe_get(1), Float64(0))
    for i in range(2):
        assert_true(not (decisions.unsafe_get(i) > 0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
