from moxie._random import RandomState
from std.testing import assert_equal

def main() raises:
    var rng = RandomState(0)
    assert_equal(rng.random_sample(), Float64(0.5488135039273248))
    print("PASS: pinned shared RandomState is importable and seeded")
