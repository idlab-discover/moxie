import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs, inf
from moxie.svm._contracts import Dense, LinearOptions
from moxie.svm._liblinear import LibLinear, _bounded
from moxie._random import RandomState


def close(a: Dense, b: Dense) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_true(abs(a.unsafe_get(i) - b.unsafe_get(i)) < 1e-10)


def main() raises:
    var engine = LibLinear()
    var x = nm.zeros[DType.float64](nm.Shape(6, 2))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    var sw = nm.ones[DType.float64](nm.Shape(6))
    var cw = nm.ones[DType.float64](nm.Shape(2))
    for i in range(6):
        x.unsafe_set(2 * i, Float64(i) - 2.5)
        x.unsafe_set(2 * i + 1, Float64(i % 2))
        y.unsafe_set(i, Float64(i // 3))
    var before = x.copy()
    for solver in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13]:
        var options = LinearOptions(solver, 1, 0.1, 1e-7, True, 2, 5000, 42)
        var classes = cw.copy()
        if solver >= 11:
            classes = nm.zeros[DType.float64](nm.Shape(0))
        var model = engine.solve(x, y, sw, classes, options)
        var repeated = engine.solve(x, y, sw, classes, options)
        close(model.coef, repeated.coef)
        close(model.intercept, repeated.intercept)
        assert_equal(model.fit_status, 0)
        assert_equal(model.coef.shape[1], 2)
        assert_equal(model.iterations.shape[0], 1)
        var expected_rows = 1
        if solver == 4:
            expected_rows = 2
        assert_equal(model.coef.shape[0], expected_rows)
        var decisions = engine.decision_values(x, model)
        assert_equal(decisions.shape[0], 6)
        assert_equal(decisions.shape[1], expected_rows)
        for i in range(6):
            for r in range(expected_rows):
                var expected = model.intercept.unsafe_get(r)
                for j in range(2):
                    expected += x.unsafe_get(2 * i + j) * model.coef.unsafe_get(r * 2 + j)
                assert_true(abs(expected - decisions.unsafe_get(i * expected_rows + r)) < 1e-12)
        # Caller mutations do not alias model coefficients or an earlier result.
        var saved = model.coef.unsafe_get(0)
        repeated.coef.unsafe_set(0, 123)
        assert_equal(model.coef.unsafe_get(0), saved)
        close(x, before)
        # Noncontiguous input and targets, including negative strides and offsets.
        var backing = nm.zeros[DType.float64](nm.Shape(6, 4))
        var targets = nm.zeros[DType.float64](nm.Shape(12))
        var sample_weights = nm.zeros[DType.float64](nm.Shape(12))
        for i in range(6):
            backing.unsafe_set(i * 4 + 3, x.unsafe_get(i * 2))
            backing.unsafe_set(i * 4 + 1, x.unsafe_get(i * 2 + 1))
            targets.unsafe_set(11 - 2 * i, y.unsafe_get(i))
            sample_weights.unsafe_set(2 * i, 1)
        var strided = backing.view_with_layout(nm.Shape(6, 2), nm.NDArrayStrides(4, -2), 3)
        var strided_y = targets.view_with_layout(nm.Shape(6), nm.NDArrayStrides(-2), 11)
        var strided_sw = sample_weights.view_with_layout(nm.Shape(6), nm.NDArrayStrides(2), 0)
        var equivalent = engine.solve(strided, strided_y, strided_sw, classes, options)
        close(model.coef, equivalent.coef)
        close(model.intercept, equivalent.intercept)
        close(decisions, engine.decision_values(strided, equivalent))
        # Options affect new solves; previous learned results stay independent.
        options.c = 0.1
        var changed = engine.solve(x, y, sw, classes, options)
        assert_equal(model.coef.unsafe_get(0), saved)
        options.max_iter = 1
        options.tol = 1e-15
        var limited = engine.solve(x, y, sw, classes, options)
        if solver != 4:
            assert_true(Int(limited.iterations.unsafe_get(0)) <= 1)
        else:
            assert_true(Int(limited.iterations.unsafe_get(0)) > 1)
        options.max_iter = 0
        if solver != 4:
            var empty_fit = engine.solve(x, y, sw, classes, options)
            assert_equal(empty_fit.iterations.unsafe_get(0), 0)
            assert_equal(empty_fit.fit_status, 1)
    var options = LinearOptions(1, 1, 0.1, 1e-4, True, 1, 100, 42)
    var short_y = nm.ones[DType.float64](nm.Shape(5))
    var empty_x = nm.ones[DType.float64](nm.Shape(0, 2))
    with assert_raises():
        _ = engine.solve(empty_x, y, sw, cw, options)
    with assert_raises():
        _ = engine.solve(x, short_y, sw, cw, options)
    with assert_raises():
        _ = engine.solve(y, y, sw, cw, options)
    var bad_y = y.copy()
    bad_y.unsafe_set(0, 0.5)
    with assert_raises():
        _ = engine.solve(x, bad_y, sw, cw, options)
    bad_y.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = engine.solve(x, bad_y, sw, cw, options)
    var zero_sw = nm.zeros[DType.float64](nm.Shape(6))
    with assert_raises():
        _ = engine.solve(x, y, zero_sw, cw, options)
    for invalid in [-1, 8, 9, 10, 14]:
        options.solver = invalid
        with assert_raises():
            _ = engine.solve(x, y, sw, cw, options)
    options.solver = 1
    options.c = 0
    with assert_raises():
        _ = engine.solve(x, y, sw, cw, options)
    options.c = 1
    options.tol = 0
    with assert_raises():
        _ = engine.solve(x, y, sw, cw, options)
    options.tol = 1e-4
    options.intercept_scaling = 0
    with assert_raises():
        _ = engine.solve(x, y, sw, cw, options)
    options.intercept_scaling = 1
    options.seed = -1
    with assert_raises():
        _ = engine.solve(x, y, sw, cw, options)
    var rng = RandomState(42)
    # newrand consumes a word even when only a single index is possible.
    assert_equal(_bounded(rng, 1), 0)
    assert_equal(rng.state_position(), 1)
    print("PASS all solver contracts, strides, ownership, limits and invalid inputs")
