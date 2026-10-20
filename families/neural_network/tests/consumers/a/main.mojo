from numojo import zeros, Shape
from moxie.probe_a import ProbeA
from moxie.probe_b import ProbeB
from moxie._random import RandomState
from std.testing import assert_equal, assert_false, assert_raises

def main() raises:
    var a = ProbeA()
    var b = ProbeB()
    var X = zeros[DType.float64](Shape(2, 2))
    X.unsafe_set(0, 3)
    X.unsafe_set(1, 4)
    with assert_raises():
        _ = a.predict(X)
    a.fit(X)
    b.fit(a.learned.value())
    X.unsafe_set(0, 9)
    assert_equal(a.learned.value().unsafe_get(0), 3)
    var result = b.predict(a.learned.value())
    assert_equal(result.unsafe_get(0), 25)
    a.fit(X)
    assert_equal(a.learned.value().unsafe_get(0), 9)
    assert_equal(b.learned.value().unsafe_get(0), 3)
    var invalid = zeros[DType.float64](Shape(0, 2))
    with assert_raises():
        a.fit(invalid)
    assert_equal(a.learned.value().unsafe_get(0), 9)
    var cloned = a.clone()
    assert_false(cloned.state.is_fitted())
    with assert_raises():
        _ = a.predict(zeros[DType.float64](Shape(1, 3)))
    var rng = RandomState(42)
    var rng_copy = rng.copy()
    assert_equal(a.draw(rng), b.draw(rng_copy))
    print("PASS isolated consumer: shared dense storage, validation and RNG")
