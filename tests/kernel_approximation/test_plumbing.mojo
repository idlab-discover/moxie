from numojo import NDArray, Item, Shape, NDArrayStrides, zeros
from moxie._kernel_features import validate_matrix, owned_snapshot, component_width, additive_width, basis_width, FeatureMapState
from moxie._random import RandomState
from std.testing import assert_equal, assert_raises


def contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(2, 3))
    x[Item(0, 0)] = 2
    validate_matrix(x, True)
    var saved = owned_snapshot(x)
    var view = x.view()
    view[Item(0, 0)] = 7
    assert_equal(x.item(0, 0), 7)
    assert_equal(saved.item(0, 0), 2)
    var state = FeatureMapState()
    with assert_raises():
        state.check_transform(x)
    state.record_fit(x, 5)
    state.check_transform(x)
    var independent = state.copy()
    state.reset()
    independent.check_transform(x)
    assert_equal(independent.n_features_out_, 5)
    var changed = zeros[dtype](Shape(1, 4))
    with assert_raises():
        independent.check_transform(changed)
    with assert_raises():
        independent.record_fit(changed, 0)
    independent.check_transform(x)
    independent.record_fit(changed, 8)
    assert_equal(independent.n_features_out_, 8)
    independent.check_transform(changed)
    x[Item(0, 1)] = -1
    validate_matrix(x)
    with assert_raises():
        validate_matrix(x, True)
    var empty = zeros[dtype](Shape(0, 3))
    with assert_raises():
        validate_matrix(empty)
    var vector = zeros[dtype](Shape(3))
    with assert_raises():
        validate_matrix(vector)
    x[Item(0, 1)] = SIMD[dtype, 1](0) / SIMD[dtype, 1](0)
    with assert_raises():
        validate_matrix(x)
    x[Item(0, 1)] = SIMD[dtype, 1](1) / SIMD[dtype, 1](0)
    with assert_raises():
        validate_matrix(x)
    var backing = zeros[dtype](Shape(3, 4))
    var strided = backing.view_with_layout(Shape(2, 2), NDArrayStrides(4, -1), 2)
    validate_matrix(strided)
    var snapshot = owned_snapshot(strided)
    strided[Item(0, 0)] = 9
    assert_equal(snapshot.item(0, 0), 0)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    assert_equal(component_width(100), 100)
    assert_equal(additive_width(3, 1), 3)
    assert_equal(additive_width(3, 2), 9)
    assert_equal(additive_width(3, 3), 15)
    assert_equal(basis_width(2, 100), 2)
    assert_equal(basis_width(100, 2), 2)
    with assert_raises():
        _ = additive_width(Int.MAX, 2)
    with assert_raises():
        _ = additive_width(1, Int.MAX)
    with assert_raises():
        _ = component_width(0)
    with assert_raises():
        _ = basis_width(0, 100)
    var rng = RandomState(42)
    var first = rng.random_sample()
    var reset = RandomState(42)
    assert_equal(first, reset.random_sample())
    var copied = rng.copy()
    assert_equal(rng.standard_normal(), copied.standard_normal())
    with assert_raises():
        _ = RandomState(-1)
    print("PASS: dense feature-map contracts")
