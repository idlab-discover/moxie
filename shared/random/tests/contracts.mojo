from moxie._random import RandomState, NativeRandom, entropy_state
from std.random import Random as Philox
from std.math import copysign
from std.testing import assert_equal, assert_true, assert_raises
import numojo as nm
from numojo import Shape, Item

def main() raises:
    var a = RandomState(42)
    var b = RandomState(42)
    for _ in range(2000):
        assert_equal(a.next_uint32(), b.next_uint32())
    var copied = a.copy()
    assert_equal(a.standard_normal(), copied.standard_normal())
    var cached = a.copy()
    assert_equal(a.standard_normal(), cached.standard_normal())
    _ = a.next_uint32()
    assert_true(a.state_position() != cached.state_position())
    var native = NativeRandom(42)
    var native_copy = native.copy()
    assert_equal(native.step(), native_copy.step())
    var context = entropy_state()
    _ = context.random_sample()
    with assert_raises():
        _ = RandomState(-1)
    with assert_raises():
        _ = RandomState(4294967296)
    with assert_raises():
        _ = a.randint(0)
    with assert_raises():
        _ = a.normal(0, -1)
    # Gamma copies retain the shared normal cache and advance independently.
    var gamma_rng = RandomState(42)
    _ = gamma_rng.standard_normal()
    var gamma_copy = gamma_rng.copy()
    for shape in [Float64(0), 0.01, 0.3, 1.0, 1.000001, 100.0]:
        for _ in range(100):
            assert_equal(gamma_rng.gamma(shape), gamma_copy.standard_gamma(shape))
    assert_equal(gamma_rng.standard_normal(), gamma_copy.standard_normal())
    for _ in range(625):
        assert_equal(gamma_rng.next_uint32(), gamma_copy.next_uint32())
    var zero = RandomState(42)
    assert_equal(zero.gamma(0.0), 0.0)
    assert_equal(zero.state_position(), 624)
    var zero_copy = zero.copy()
    assert_equal(zero.gamma(100.0, 0.0), 0.0)
    _ = zero_copy.gamma(100.0)
    assert_equal(zero.standard_normal(), zero_copy.standard_normal())
    for _ in range(625):
        assert_equal(zero.next_uint32(), zero_copy.next_uint32())
    var negative_permutation = a.permutation(-1)
    assert_equal(negative_permutation.size, 0)
    with assert_raises():
        _ = a.binomial(-1, 0.5)
    with assert_raises():
        _ = a.binomial(10, 1.1)
    var p = nm.zeros[DType.float64](Shape(3))
    with assert_raises():
        _ = a.categorical(p)
    p[Item(0)] = 1.0
    for _ in range(20):
        assert_equal(a.categorical(p), 0)
    for method in [String("auto"), "pool", "tracking_selection", "reservoir_sampling"]:
        with assert_raises():
            _ = a.sample_without_replacement(2, 3, method)
        with assert_raises():
            _ = a.sample_without_replacement(2, -1, method)
        var empty = a.sample_without_replacement(0, 0, method)
        assert_equal(empty.size, 0)
        for k in range(21):
            var sample = a.sample_without_replacement(20, k, method)
            assert_equal(sample.size, k)
            for i in range(k):
                assert_true(Int(sample.item(i)) >= 0 and Int(sample.item(i)) < 20)
                for j in range(i):
                    assert_true(sample.item(i) != sample.item(j))
    with assert_raises():
        _ = a.sample_without_replacement(0, 0, "invalid")
    # Copy with a populated Gaussian cache, then reseed a separate instance.
    var first = RandomState(12)
    _ = first.standard_normal()
    assert_true(first.has_gauss())
    var second = first.copy()
    assert_equal(first.standard_normal(), second.standard_normal())
    for _ in range(1300):
        assert_equal(first.next_uint32(), second.next_uint32())
    var other = RandomState(13)
    assert_true(first.next_uint32() != other.next_uint32())
    var philox = Philox(seed=UInt64(42))
    var adapter = NativeRandom(42)
    for _ in range(10):
        assert_equal(philox.step(), adapter.step())
    var before = a.copy()
    with assert_raises():
        _ = a.uniform(Float64.MAX, -Float64.MAX)
    with assert_raises():
        _ = a.state_word(-1)
    with assert_raises():
        _ = a.state_word(624)
    with assert_raises():
        _ = a.multinomial(-1, p)
    with assert_raises():
        _ = a.multinomial(1, p, -1)
    var badrank = nm.zeros[DType.float64](Shape(1, 3))
    with assert_raises():
        _ = a.categorical(badrank)
    p[Item(0)] = 0.8
    p[Item(1)] = 0.8
    with assert_raises():
        _ = a.multinomial(1, p)
    p[Item(0)] = -0.1
    with assert_raises():
        _ = a.categorical(p)
    p[Item(0)] = Float64.MAX * 2.0
    with assert_raises():
        _ = a.categorical(p)
    p[Item(0)] = p.item(0) - p.item(0)
    with assert_raises():
        _ = a.multinomial(1, p)
    var infinity = Float64.MAX * 2.0
    var negative_zero = copysign(Float64(0), Float64(-1))
    for invalid in [Float64(-1), negative_zero, infinity, -infinity, infinity - infinity]:
        with assert_raises():
            _ = a.standard_gamma(invalid)
        with assert_raises():
            _ = a.gamma(invalid)
        with assert_raises():
            _ = a.gamma(100.0, invalid)
    # All validation failures precede consumption.
    for _ in range(625):
        assert_equal(a.next_uint32(), before.next_uint32())
    p[Item(0)] = 0.2
    p[Item(1)] = 0.3
    p[Item(2)] = 0.0  # NumPy's final probability is residual.
    var result = a.multinomial(100, p, 17)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 17)
    assert_equal(result.shape[1], 3)
    for i in range(17):
        assert_equal(result.item(i,0) + result.item(i,1) + result.item(i,2), 100)
    test_choice_shuffle()
    print("PASS native random contracts")


def test_choice_shuffle() raises:
    var rng = RandomState(42)
    var expected = rng.copy()
    var a = nm.zeros[DType.float32](Shape(10))
    for i in range(10):
        a[Item(i)] = Float32(i + 10)
    var p = nm.zeros[DType.float64](Shape(10))
    for i in range(10):
        p[Item(i)] = 0.1
    var values = rng.choice(a, p, 5, False)
    var indices = expected.choice(10, p, 5, False)
    for i in range(5):
        assert_equal(values.item(i), a.item(Int(indices.item(i))))
    values = rng.choice(a, 15)
    indices = expected.choice(10, 15)
    for i in range(15):
        assert_equal(values.item(i), a.item(Int(indices.item(i))))
    # Negative-stride views mutate only the selected backing cells.
    var view = a.view_with_layout(Shape(5), nm.NDArrayStrides(-2), 8)
    var order = expected.permutation(5)
    rng.shuffle(view)
    for i in range(5):
        assert_equal(view.unsafe_get(-2*i), Float32(18 - 2 * Int(order.item(i))))
        assert_equal(a.item(2*i+1), Float32(11+2*i))
    var vector_state = rng.copy()
    var chosen = rng.choice(view, 7)
    var chosen_indices = vector_state.choice(5, 7)
    for i in range(7):
        assert_equal(chosen.item(i), view.unsafe_get(-2*Int(chosen_indices.item(i))))
    var probability_backing = nm.zeros[DType.float64](Shape(20))
    for i in range(10):
        probability_backing[Item(2*i)] = p.item(i)
    var probability_view = probability_backing.view_with_layout(Shape(10), nm.NDArrayStrides(2), 0)
    indices = rng.choice(10, probability_view, 6, False)
    var expected_indices = vector_state.choice(10, p, 6, False)
    for i in range(6):
        assert_equal(indices.item(i), expected_indices.item(i))
    var copied = rng.permutation(view)
    assert_equal(copied.ndim, 1)
    copied[Item(0)] = -99
    for i in range(5):
        assert_true(view.unsafe_get(-2*i) != -99)
    # Transposed matrix: shuffle columns of the original backing matrix.
    var matrix = nm.zeros[DType.int64](Shape(3, 4))
    for i in range(12):
        matrix.unsafe_set(i, Int64(i))
    var transpose = matrix.view_with_layout(Shape(4, 3), nm.NDArrayStrides(1, 4), 0)
    var snapshot = rng.copy()
    order = snapshot.permutation(4)
    rng.shuffle(transpose)
    for row in range(4):
        for col in range(3):
            assert_equal(transpose.item(row,col), Int64(4*col+Int(order.item(row))))
    assert_equal(rng.next_uint32(), snapshot.next_uint32())
    snapshot = rng.copy()
    with assert_raises():
        _ = rng.choice(0)
    with assert_raises():
        _ = rng.choice(3, -1)
    with assert_raises():
        _ = rng.choice(3, 4, False)
    with assert_raises():
        _ = rng.choice(3, p)
    with assert_raises():
        _ = rng.choice(matrix)
    p[Item(0)] = 1
    for i in range(1,10):
        p[Item(i)] = 0
    with assert_raises():
        _ = rng.choice(10, p, 2, False)
    var rank3 = nm.zeros[DType.float64](Shape(2,2,2))
    with assert_raises():
        rng.shuffle(rank3)
    var overlap = matrix.view_with_layout(Shape(3, 3), nm.NDArrayStrides(1, 1), 0)
    with assert_raises():
        rng.shuffle(overlap)
    assert_equal(rng.next_uint32(), snapshot.next_uint32())
