from std.testing import assert_equal, assert_false
import numojo as nm
from moxie.probe_a import snapshot, measure, EstimatorProbe as ProbeA
from moxie.probe_b import measure as measure_b, EstimatorProbe as ProbeB

def exercise[dtype: DType]() raises:
    var x = nm.ones[dtype](nm.Shape(2, 3))
    var a = ProbeA()
    var b = ProbeB()
    a.fit(x)
    b.fit(x)
    assert_equal(a.signature_, b.signature_)
    var clone = a.__sklearn_clone__()
    assert_false(clone.fitted)
    var output = snapshot(x)
    assert_equal(measure(x), measure_b(output))
    x.unsafe_set(0, 5)
    assert_equal(output.embedding.unsafe_get(0), 1)
    var view = output.embedding.view()
    view.unsafe_set(0, 2)
    assert_equal(output.embedding.unsafe_get(0), 2)
    assert_equal(measure(output.embedding), measure_b(output))

def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("PASS: isolated composed namespace and shared storage")
