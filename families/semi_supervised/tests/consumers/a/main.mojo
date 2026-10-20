from std.testing import assert_equal
from moxie.semi_supervised import produce, consume
from moxie.semi_supervised._contracts import ClassEncoding, integer_targets

def main() raises:
    var values = produce()
    assert_equal(consume(values), 3.0)
    var view = values.view()
    view.unsafe_set(0, 5)
    assert_equal(consume(values), 5.0)
    var codec = ClassEncoding(List[Int]([-7, 2]))
    var codes = codec.encode(integer_targets(List[Int]([-1, 2, -7])))
    assert_equal(codes[0], -1)
    assert_equal(codes[1], 1)
    print("isolated consumer passed")
