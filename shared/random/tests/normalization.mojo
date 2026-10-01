from moxie._random import (
    RandomState, RandomStateHandle, check_random_state, seed_global_random_state,
)
from std.testing import assert_equal, assert_raises, assert_true
import std.random as native_random
from max.algorithm import parallelize
import numojo as nm
from numojo import Shape


def same(a: RandomState, b: RandomState) raises:
    assert_equal(a.state_position(), b.state_position())
    assert_equal(a.has_gauss(), b.has_gauss())
    assert_equal(a.cached_gauss(), b.cached_gauss())
    for i in range(624):
        assert_equal(a.state_word(i), b.state_word(i))


def moved_draw(var handle: RandomStateHandle[MutUntrackedOrigin]) -> Float64:
    return handle[].standard_normal()


def main() raises:
    # First use needs no explicit seed and initializes only once.
    var snapshot = RandomState(0)
    with check_random_state() as rng:
        var value = rng[].random_sample()
        assert_true(value >= 0 and value < 1)
        snapshot = rng[].copy()
    with check_random_state(None) as rng:
        same(rng[], snapshot)
        assert_equal(rng[].random_sample(), snapshot.random_sample())

    seed_global_random_state(901)
    var expected_global = RandomState(901)
    for _ in range(3):
        with check_random_state(42) as rng:
            var expected = RandomState(42)
            assert_equal(rng[].standard_normal(), expected.standard_normal())
            same(rng[], expected)
        with check_random_state(Optional[Int](42)) as rng:
            var expected = RandomState(42)
            assert_equal(rng[].random_sample(), expected.random_sample())
    with check_random_state() as rng:
        same(rng[], expected_global)

    var borrowed = RandomState(42)
    _ = borrowed.standard_normal()
    var expected = borrowed.copy()
    for _ in range(3):
        with check_random_state(borrowed) as rng:
            same(rng[], expected)
            assert_equal(rng[].standard_normal(), expected.standard_normal())
            assert_equal(rng[].random_sample(), expected.random_sample())
        same(borrowed, expected)

    for _ in range(3):
        with check_random_state(Optional[Int]()) as rng:
            same(rng[], expected_global)
            # Nested consumers borrow the active stream, without reacquiring it.
            with check_random_state(rng[]) as nested:
                assert_equal(nested[].standard_normal(), expected_global.standard_normal())
            with assert_raises():
                _ = check_random_state()
            with assert_raises():
                seed_global_random_state(42)
            # Failed acquire must not release somebody else's lease.
            with assert_raises():
                _ = check_random_state(None)
            same(rng[], expected_global)

    # Exception unwinding releases the lease and preserves consumed state.
    try:
        with check_random_state() as rng:
            assert_equal(rng[].standard_normal(), expected_global.standard_normal())
            raise Error("consumer failed after a draw")
    except error:
        assert_equal(String(error), "consumer failed after a draw")
    with check_random_state() as rng:
        same(rng[], expected_global)
    var handle = check_random_state()
    assert_equal(moved_draw(handle^), expected_global.standard_normal())
    with check_random_state() as rng:
        same(rng[], expected_global)
    for bad in [-1, 4294967296]:
        with assert_raises():
            _ = check_random_state(bad)
        with assert_raises():
            _ = check_random_state(Optional[Int](bad))
        with assert_raises():
            seed_global_random_state(bad)
    with check_random_state() as rng:
        same(rng[], expected_global)
    seed_global_random_state(42)
    with check_random_state() as rng:
        var fresh = RandomState(42)
        same(rng[], fresh)
        assert_true(not rng[].has_gauss())

    # The new singleton must not seed or consume std.random's Philox singleton.
    native_random.seed(123)
    var first = native_random.random_float64()
    var second = native_random.random_float64()
    native_random.seed(123)
    assert_equal(native_random.random_float64(), first)
    seed_global_random_state(99)
    with check_random_state() as rng:
        _ = rng[].standard_normal()
    assert_equal(native_random.random_float64(), second)
    concurrent_rejection()
    print("PASS normalized seed/borrow/global ownership, cache, reset, moves, exceptions and concurrent rejection")


def concurrent_rejection() raises:
    var results = nm.zeros[DType.int64](Shape(32))
    var target = results.unsafe_ptr()
    with check_random_state() as active:
        var before = active[].copy()
        def worker(i: Int) {target}:
            try:
                if i % 2 == 0:
                    seed_global_random_state(42)
                else:
                    with check_random_state() as rng:
                        _ = rng[].random_sample()
                target[unsafe_offset=i] = 1
            except error:
                target[unsafe_offset=i] = 2 if String(error).startswith("RuntimeError: global RandomState is already borrowed") else 3
        parallelize(worker, 32, 4)
        same(active[], before)
    for i in range(32):
        assert_equal(results.item(i), 2)
