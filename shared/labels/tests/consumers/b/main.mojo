from moxie._labels import StringTarget, encode_column
from std.testing import assert_equal
from numojo import Shape, Item
import numojo as nm


def main() raises:
    # Splitter-shaped first-appearance IDs and nonintegral group identity.
    var y = StringTarget(["z", "a", "z", "b"])
    var encoded = encode_column(y)
    var codes = encoded.index.first_seen_codes()
    assert_equal(codes[Item(0)], Int64(0))
    assert_equal(codes[Item(1)], Int64(1))
    assert_equal(codes[Item(3)], Int64(2))
    var groups = nm.zeros[DType.float64](Shape(3))
    groups[Item(0)] = 1.25
    groups[Item(1)] = 1.5
    groups[Item(2)] = 1.25
    var group_ids = encode_column(groups, classification=False)
    assert_equal(group_ids.index.counts[Item(0)], Int64(2))
    assert_equal(group_ids.classes[Item(1)], Float64(1.5))
    print("PASS independent splitter consumer")
