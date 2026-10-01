from moxie._labels import StringTarget, encode_column
from std.testing import assert_equal
from numojo import Item


def main() raises:
    # Classifier-shaped lookup: independent domains and dtype-preserving decode.
    var y = StringTarget(["z", "north", "a", "south", "z", "north"], 3, 2)
    var domain = encode_column(y, 0)
    var second = encode_column(y, 1)
    var prediction = domain.decode(domain.index.inverse)
    assert_equal(prediction.get(0), "z")
    assert_equal(domain.classes[0], "a")
    assert_equal(second.classes[0], "north")
    print("PASS independent classifier consumer")
