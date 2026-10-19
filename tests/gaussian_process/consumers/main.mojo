"""Both separately owned consumers call one assembled shared implementation."""
import numojo as nm
from consumer_a import ConsumerA
from consumer_b import ConsumerB
from moxie._random import RandomState
from std.testing import assert_equal

def check[dtype: DType]() raises:
    var X = nm.arange[dtype](6).reshape(nm.Shape(2, 3))
    var a = ConsumerA()
    var b = ConsumerB()
    var rng = RandomState(42)
    var expected_rng = rng.copy()
    var first = a.inspect(X, rng)
    var second = b.inspect(X, rng)
    assert_equal(first.item(0), 5)
    assert_equal(first.item(1), 50)
    assert_equal(second.item(1), first.item(1))
    _ = expected_rng.random_sample()
    _ = expected_rng.random_sample()
    assert_equal(rng.next_uint32(), expected_rng.next_uint32())

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: both consumers share dense storage, GP retention and RNG")
