from moxie import evaluate_a
from moxie._dense import sum_axis2d
from moxie.utils import FittedState
from numojo import NDArray, Shape, ones
from std.testing import assert_equal, assert_raises


def probe[dtype: DType]() raises:
    var data = ones[dtype](Shape(2, 3))
    assert_equal(evaluate_a(data), 3.0)
    var shared = data.view()
    shared.unsafe_set(0, 2)
    assert_equal(evaluate_a(data), 6.0)
    var independent = data.copy()
    independent.unsafe_set(0, 9)
    assert_equal(evaluate_a(data), 6.0)
    var sums = sum_axis2d(data, 1)
    assert_equal(Float64(sums.item(0)), 4.0)
    with assert_raises():
        _ = sum_axis2d(data, 9)
    print(evaluate_a(data), sums.item(0))


def main() raises:
    var state = FittedState()
    with assert_raises():
        state.check_features(3)
    with assert_raises():
        state.record_fit(0, 3)
    state.record_fit(2, 3)
    with assert_raises():
        state.check_features(2)
    state.reset()
    assert_equal(state.is_fitted(), False)
    probe[DType.float32]()
    probe[DType.float64]()
