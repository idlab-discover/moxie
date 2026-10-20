from moxie.centroid_consumer_a import assign_samples
from moxie.centroid_consumer_b import update_samples
from std.testing import assert_equal
import numojo as nm

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(2, 1))
    x[nm.Item(1, 0)] = 4
    var c = nm.zeros[DType.float64](nm.Shape(1, 1))
    var w = nm.ones[DType.float64](nm.Shape(2))
    var a = assign_samples(x, c, w)
    var u = update_samples(x, a.labels, w, c)
    assert_equal(u.centers.item(0, 0), 2)
    assert_equal(u.masses.item(0), 2)
    var again = assign_samples(x, u.centers, w)
    assert_equal(again.inertia, 8)
    u.centers[nm.Item(0, 0)] = 10
    assert_equal(c.item(0, 0), 0)
    print("shared centroid engine consumer passed")
