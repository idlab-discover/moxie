from numojo import arange, Shape, Item
from moxie import ProbeA, ProbeB
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from std.testing import assert_equal

def check[dtype: DType]() raises:
    var x = arange[dtype](6).reshape(Shape(3, 2))
    var a = ProbeA.capture(x)
    var b = ProbeB.capture(a)
    a[Item(0, 0)] = 99
    assert_equal(x.item(0, 0), 0)
    assert_equal(b.item(0, 0), 0)
    var norms = row_squared_norms(b)
    assert_equal(norms.item(0), 1)
    var rng = RandomState(42)
    assert_equal(rng.next_uint32(), 1608637542)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS isolated shared mixture consumers")
