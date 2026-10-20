from moxie.ensemble import consumer_a, consumer_b
from moxie.ensemble.consumer_a import produce
from moxie.ensemble.consumer_b import touch
from std.testing import assert_equal

def main() raises:
    assert_equal(consumer_a(), 8.0)
    assert_equal(consumer_b(), 8.0)
    var values = produce()
    var shared = values.view()
    touch(shared)
    assert_equal(values.item(0, 0), 17)
    print("Both contributors share the same ensemble, dense and RNG implementations")
