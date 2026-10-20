import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.linear_model.probe_a import ConsumerEstimator as First
from moxie.linear_model.probe_b import ConsumerEstimator as Second
from moxie._dense import row_squared_norms
from moxie._random import RandomState

def main() raises:
    var X = nm.ones[DType.float64](nm.Shape(2, 3))
    var y = nm.zeros[DType.float64](nm.Shape(2))
    var a = First()
    var b = Second()
    with assert_raises():
        _ = a.first()
    a.fit(X, y)
    b.fit(X, y)
    X.unsafe_set(0, 9)
    assert_equal(a.first(), 1)
    assert_equal(b.first(), 1)
    a.fit(X, y)
    assert_equal(a.first(), 9)
    assert_equal(b.first(), 1)
    with assert_raises():
        a.fit(X, nm.zeros[DType.float64](nm.Shape(3)))
    assert_equal(a.first(), 9)
    var norms = row_squared_norms(X)
    assert_equal(norms.unsafe_get(0), 83)
    var rng = RandomState(42)
    assert_equal(rng.next_uint32(), UInt32(1608637542))
    print("isolated shared plumbing consumer passed")
