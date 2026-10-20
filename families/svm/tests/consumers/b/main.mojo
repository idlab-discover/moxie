import numojo as nm
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.svm import ConsumerA, ConsumerB, LinearSVC, LinearSVR
from moxie.base import clone
from moxie._random import RandomState

def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(2, 3))
    var y = nm.zeros[DType.float64](nm.Shape(2))
    var a = ConsumerA(7)
    var b = ConsumerB(7)
    _ = a.fit(x, y)
    _ = b.fit(x, y)
    assert_equal(a.norms.unsafe_get(0), 3.0)
    assert_equal(b.norms.unsafe_get(0), a.norms.unsafe_get(0))
    x.unsafe_set(0, 8)
    assert_equal(a.norms.unsafe_get(0), 3.0)
    var fresh_a = clone(a)
    var fresh_b = clone(b)
    assert_false(fresh_a.__sklearn_is_fitted__())
    assert_false(fresh_b.__sklearn_is_fitted__())
    assert_equal(fresh_a.get_params().marker, 7)
    assert_equal(fresh_b.get_params().marker, 7)
    assert_equal(fresh_a.norms.size, 0)
    var params = a.get_params()
    params.marker = 9
    _ = a.set_params(params)
    assert_equal(b.get_params().marker, 7)
    _ = a.fit(x, y)
    _ = b.fit(x, y)
    assert_equal(a.norms.unsafe_get(0), 66.0)
    assert_equal(b.norms.unsafe_get(0), 66.0)
    var bad_y = nm.zeros[DType.float64](nm.Shape(1))
    with assert_raises():
        _ = a.fit(x, bad_y)
    with assert_raises():
        _ = b.fit(x, bad_y)
    assert_true(a.__sklearn_is_fitted__())
    assert_equal(a.norms.unsafe_get(0), 66.0)
    var rng = RandomState(42)
    assert_equal(rng.next_uint32(), UInt32(1608637542))
    print("PASS composed SVM consumers")
    var labels: List[Float64] = [0,1]
    var classifier = LinearSVC(random_state=42)
    _ = classifier.fit(x, labels)
    assert_equal(classifier.predict(x)[0], 0.0)
    var regression = LinearSVR(random_state=42)
    _ = regression.fit(x, y)
    assert_equal(regression.predict(x).shape[0], 2)
