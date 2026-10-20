"""Dependency probe only: no SGD implementation or replacement RNG."""
from moxie._random import RandomState
from moxie.linear_model._contracts import validate_xy, validate_weights, snapshot
from std.testing import assert_equal
import numojo as nm


def check_storage[dtype: DType]() raises:
    var X = nm.ones[dtype](nm.Shape(2, 1))
    var y = nm.zeros[dtype](nm.Shape(2))
    var weights = nm.ones[dtype](nm.Shape(2))
    validate_xy(X, y)
    validate_weights(weights, 2)
    var independent = snapshot(X)
    X.unsafe_set(0, 9)
    assert_equal(independent.unsafe_get(0), 1)


def main() raises:
    check_storage[DType.float32]()
    check_storage[DType.float64]()
    for index in range(3):
        var seed = index
        if index == 2:
            seed = 42
        var rng = RandomState(seed)
        var independent = rng.copy()
        var first = rng.next_uint32()
        assert_equal(first, independent.next_uint32())
        print("word", seed, first)
        var order_rng = RandomState(seed)
        var order = order_rng.permutation(3)
        print("order", seed, order.unsafe_get(0), order.unsafe_get(1), order.unsafe_get(2))
