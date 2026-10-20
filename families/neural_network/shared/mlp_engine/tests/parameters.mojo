from std.math import inf
from std.testing import assert_equal, assert_true, assert_raises
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie._random import RandomState
from moxie.neural_network._plumbing import LayerLayout, LayerTensors
from moxie.neural_network._mlp_engine import allocate, initialize, pack, unpack


def check[dtype: DType]() raises:
    var layout = LayerLayout([2, 3, 1])
    var params = allocate[dtype](layout)
    var rng = RandomState(42)
    initialize(params, rng, "relu")
    var packed = pack(params)
    assert_equal(packed.shape[0], 13)
    var other = allocate[dtype](layout)
    unpack(packed, other)
    var again = pack(other)
    for i in range(13):
        assert_equal(again.unsafe_get(i), packed.unsafe_get(i))
    other.coefs[0].unsafe_set(0, 99)
    assert_true(params.coefs[0].unsafe_get(0) != 99)
    assert_true(packed.unsafe_get(0) != 99)
    var wrong = zeros[DType.float64](Shape(12))
    with assert_raises():
        unpack(wrong, other)
    assert_equal(other.coefs[0].unsafe_get(0), 99)
    with assert_raises():
        initialize(other, rng, "softmax")
    # A rejected initializer must not consume the RNG stream.
    var rng_snapshot = rng.copy()
    with assert_raises():
        initialize(other, rng, "invalid")
    assert_equal(rng.random_sample(), rng_snapshot.random_sample())
    var bad_values = packed.copy()
    bad_values.unsafe_set(12, inf[DType.float64]())
    with assert_raises():
        unpack(bad_values, other)
    assert_equal(other.coefs[0].unsafe_get(0), 99)
    comptime if dtype == DType.float32:
        bad_values.unsafe_set(12, 1e100)
        with assert_raises():
            unpack(bad_values, other)
        assert_equal(other.coefs[0].unsafe_get(0), 99)
    var wrong_rank = zeros[DType.float64](Shape(1, 13))
    with assert_raises():
        unpack(wrong_rank, other)
    var rng2 = RandomState(42)
    initialize(other, rng2, "relu")
    again = pack(other)
    for i in range(13):
        assert_equal(again.unsafe_get(i), packed.unsafe_get(i))
    # Nonzero offset and negative stride coefficient storage.
    var data = zeros[dtype](Shape(8))
    for i in range(8):
        data.unsafe_set(i, SIMD[dtype, 1](i))
    var coef = data.view_with_layout(Shape(2, 3), NDArrayStrides(-3, 1), 4)
    var bias = zeros[dtype](Shape(3))
    var cs = List[NDArray[dtype]]()
    var bs = List[NDArray[dtype]]()
    cs.append(coef^)
    bs.append(bias^)
    var strided = LayerTensors[dtype](LayerLayout([2, 3]), cs, bs)
    var vector = pack(strided)
    assert_equal(vector.unsafe_get(0), 4)
    assert_equal(vector.unsafe_get(3), 1)
    unpack(vector, strided)
    var vector2 = pack(strided)
    for i in range(9):
        assert_equal(vector.unsafe_get(i), vector2.unsafe_get(i))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS parameter initialization/packing contracts")
