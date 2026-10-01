from consumers.producer import data
from consumers.consumer import consume
from std.testing import assert_equal


def main() raises:
    var owner = data()
    var shared = owner.view()
    assert_equal(consume(shared), 6)
    assert_equal(owner.item(0, 0), 2)
    var independent = owner.copy()
    independent.unsafe_set(0, 3)
    assert_equal(owner.item(0, 0), 2)
    print("PASS: producer/consumer mutation aliases only with an explicit view")
