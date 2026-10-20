import numojo as nm
from moxie.svm import SVC, ClassWeight
from std.math import inf, nan
from std.testing import assert_raises, assert_equal
from moxie.svm._classification import Gamma, Probability, checked_int32

def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(2, 2))
    var gamma = Gamma("value")
    with assert_raises():
        _ = gamma.resolve(x, "rbf")
    var probability = Probability("true")
    with assert_raises():
        _ = probability.enabled()
    var enabled = Probability(True)
    assert_equal(enabled.enabled(), True)
    var metadata = nm.zeros[DType.int64](nm.Shape(1))
    metadata.unsafe_set(0, 2147483648)
    with assert_raises():
        _ = checked_int32(metadata)
    var model = SVC[Float32]()
    var fractional: List[Float32] = [0.1, 0.9]
    with assert_raises():
        _ = model.fit(x, fractional)
    fractional[1] = 1
    fractional[0] = inf[DType.float32]()
    with assert_raises():
        _ = model.fit(x, fractional)
    fractional[0] = nan[DType.float32]()
    with assert_raises():
        _ = model.fit(x, fractional)
    var discrete: List[Float32] = [0, 1]
    _ = model.fit(x, discrete)
    assert_equal(model.classes_[1], Float32(1))
    var labels: List[Int] = [0, 1]
    var incomplete = SVC[Int](class_weight=ClassWeight[Int]([0, 99], [1.0, 2.0]))
    with assert_raises():
        _ = incomplete.fit(x, labels)
    var partial = SVC[Int](class_weight=ClassWeight[Int]([0], [2.0]))
    _ = partial.fit(x, labels)
    assert_equal(partial.class_weight_.unsafe_get(0), 2.0)
    assert_equal(partial.class_weight_.unsafe_get(1), 1.0)
    var extra = SVC[Int](class_weight=ClassWeight[Int]([0, 1, 99], [2.0, 1.0, 3.0]))
    _ = extra.fit(x, labels)
    assert_equal(extra.class_weight_.unsafe_get(0), 2.0)
