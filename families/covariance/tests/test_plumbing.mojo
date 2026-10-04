import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.covariance._state import CovarianceState, validate_matrix
from moxie.covariance._engines import CenteredData, ShrinkageStatistics, RobustCovarianceState
from moxie._dense import sum_axis2d, row_squared_norms


def exercise[dtype: DType]() raises:
    var x = nm.ones[dtype](nm.Shape(2, 2))
    var location = nm.zeros[dtype](nm.Shape(2))
    var state = CovarianceState[dtype]()
    with assert_raises():
        state.check_input(x)
    state.replace(location, x, x.copy())
    state.check_input(x)
    x.unsafe_set(0, 9)
    assert_equal(state.covariance_.value().unsafe_get(0), 1)
    assert_equal(state.precision_.value().unsafe_get(0), 1)
    var copied = state.copy()
    state.covariance_.value().unsafe_set(0, 8)
    assert_equal(copied.covariance_.value().unsafe_get(0), 1)
    state.reset()
    assert_false(state.fitted.is_fitted())
    assert_true(copied.fitted.is_fitted())
    assert_equal(copied.covariance_.value().unsafe_get(0), 1)
    copied.replace(location, x)
    assert_false(Bool(copied.precision_))
    var bad = nm.ones[dtype](nm.Shape(2, 3))
    with assert_raises():
        copied.replace(location, bad)
    assert_equal(copied.covariance_.value().unsafe_get(0), 9)
    with assert_raises():
        copied.replace(location, x, bad.copy())
    with assert_raises():
        copied.replace(nm.zeros[dtype](nm.Shape(3)), x)
    var bad_location = location.copy()
    bad_location.unsafe_set(0, Float64("inf").cast[dtype]())
    with assert_raises():
        copied.replace(bad_location, x)
    assert_equal(copied.covariance_.value().unsafe_get(0), 9)
    with assert_raises():
        copied.check_input(bad)
    with assert_raises():
        validate_matrix(location)
    with assert_raises():
        validate_matrix(nm.zeros[dtype](nm.Shape(0, 2)))
    with assert_raises():
        validate_matrix(nm.zeros[dtype](nm.Shape(2, 0)))
    with assert_raises():
        validate_matrix(x, 3)
    var nonfinite = nm.ones[dtype](nm.Shape(2, 2))
    nonfinite.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises():
        validate_matrix(nonfinite)
    nonfinite.unsafe_set(0, Float64("inf").cast[dtype]())
    with assert_raises():
        copied.replace(location, nonfinite)
    var view = x.view_with_layout(nm.Shape(2, 2), nm.NDArrayStrides(-2, 1), 2)
    validate_matrix(view)
    copied.replace(location, view)
    view.unsafe_set(0, 7)
    assert_equal(copied.covariance_.value().unsafe_get(0), 1)
    var c = CenteredData[dtype](x.copy(), location.copy())
    var stats = ShrinkageStatistics[dtype](x.copy(), 2, 2, 4, 8)
    var robust = RobustCovarianceState[dtype](copied.copy(), location.copy(), x.copy(),
        nm.zeros[DType.bool](nm.Shape(2)), nm.zeros[DType.bool](nm.Shape(2)),
        nm.zeros[dtype](nm.Shape(2)))
    assert_equal(stats.n_samples, 2)
    assert_equal(c.location.shape[0], 2)
    assert_equal(robust.dist_.shape[0], 2)
    var sums = sum_axis2d(x, 0)
    var norms = row_squared_norms(x)
    assert_equal(sums.shape[0], 2)
    assert_equal(norms.shape[0], 2)
    var singleton = nm.ones[dtype](nm.Shape(1, 1))
    copied.replace(nm.zeros[dtype](nm.Shape(1)), singleton)
    assert_equal(copied.fitted.n_features_in_.value(), 1)


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("PASS: dense covariance ownership and validation")
