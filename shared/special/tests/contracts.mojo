from std.testing import assert_equal, assert_true, assert_raises
from std.math import sqrt
from numax.core.libm import exp, erf
from moxie._special import gammainc


def main() raises:
    assert_equal(gammainc(Float64(0.5), Float64(0)), 0.0)
    assert_equal(gammainc(Float64(2), Float64("inf")), 1.0)
    assert_equal(gammainc(Float64(0.5), Float64(100)), 1.0)
    assert_equal(gammainc(Float32(0.5), Float32(100)), Float32(1))
    for a in [Float64(-1), 0.0, Float64("nan"), Float64("inf")]:
        with assert_raises():
            _ = gammainc(a, Float64(1))
        with assert_raises():
            _ = gammainc(Float32(a), Float32(1))
    for x in [Float64(-1), Float64("nan"), Float64("-inf")]:
        with assert_raises():
            _ = gammainc(Float64(1), x)
        with assert_raises():
            _ = gammainc(Float32(1), Float32(x))
    with assert_raises():
        _ = gammainc(Float64(1e30), Float64(1e30))
    for x in [Float64(1e-10), 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        assert_true(abs(gammainc(Float64(0.5), x) - erf(sqrt(x))) < 3e-14)
        assert_true(abs(gammainc(Float64(2), x) - (1 - exp(-x) * (1 + x))) < 3e-14)
    print("PASS gamma domain, endpoints, identities and explicit nonconvergence")
