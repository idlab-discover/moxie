from numojo import arange, Shape, Item
from moxie.probe_a import ProbeA
from moxie.probe_b import ProbeB
from std.testing import assert_equal, assert_raises
from moxie._random import RandomState

def check[dtype: DType]() raises:
    var x = arange[dtype](6).reshape(Shape(2, 3))
    var a = ProbeA[dtype]()
    var b = ProbeB[dtype]()
    with assert_raises():
        _ = a.norms()
    with assert_raises():
        _ = b.norms()
    a.fit(x)
    b.fit(a.data)
    a.data[Item(0, 0)] = 10
    assert_equal(x.item(0, 0), 0)
    assert_equal(b.norms().item(0), 5)
    a.fit(x)
    assert_equal(a.norms().item(0), b.norms().item(0))
    var shared = a.data.view()
    shared[Item(0, 0)] = 8
    assert_equal(a.data.item(0, 0), 8)
    assert_equal(b.data.item(0, 0), 0)
    var rng = RandomState(42)
    assert_equal(rng.next_uint32(), 1608637542)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: composed dense consumers")
