from moxie._special import gammainc
from std.testing import assert_equal

def main() raises:
    assert_equal(gammainc(Float64(0.5), Float64(100)), 1.0)
