from moxie import ConsumerA, ConsumerB
from moxie._random import RandomState
import numojo as nm
from std.testing import assert_equal

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(1, 2))
    x[nm.Item(0, 0)] = 3
    x[nm.Item(0, 1)] = 4
    var a = ConsumerA()
    var b = ConsumerB()
    var owned = a.snapshot(x)
    var view = owned.view()
    view[nm.Item(0, 0)] = 0
    var norms = b.norms(owned)
    assert_equal(norms.item(0), 16)
    assert_equal(x.item(0, 0), 3)
    var state = RandomState(42)
    var expected = RandomState(42)
    assert_equal(a.draw(state), expected.next_uint32())
    assert_equal(b.draw(state), expected.next_uint32())
    print("isolated cluster consumer passed")
