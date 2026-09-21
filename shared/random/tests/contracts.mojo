from moxie._random import RandomState, NativeRandom, entropy_state
from std.random import Random as Philox
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
    print("PASS native random contracts")
