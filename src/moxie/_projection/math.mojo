"""Accurate scalar logarithm for the JL integer bound."""
from std.memory import bitcast


def log64(value: Float64) -> Float64:
    """Logarithm for nonnegative finite inputs using binary range reduction.

    Mojo 1.0.0's std.math.log(100.0) differs from the reference by 2.6e-10,
    enough to change the JL integer bound for small eps. With mantissa reduced
    to [sqrt(1/2), sqrt(2)], the atanh series has |z| <= 0.172. Fourteen terms
    put its truncation error below 1e-23; arithmetic remains in Float64.
    """
    if value == 0:
        return -Float64.MAX
    var normalized = value
    var adjustment = 0
    if normalized < 2.2250738585072014e-308:
        normalized *= 18014398509481984.0
        adjustment = -54
    var bits = bitcast[DType.uint64](normalized)
    var exponent = Int((bits >> 52) & 2047) - 1023 + adjustment
    var mantissa_bits = (bits & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(
        0x3FF0000000000000
    )
    var mantissa = bitcast[DType.float64](mantissa_bits)
    if mantissa > 1.4142135623730951:
        mantissa *= 0.5
        exponent += 1
    var z = (mantissa - 1.0) / (mantissa + 1.0)
    var squared = z * z
    var term = z
    var total = z
    for k in range(1, 14):
        term *= squared
        total += term / Float64(2 * k + 1)
    return Float64(exponent) * 0.6931471803691238 + (
        2.0 * total + Float64(exponent) * 1.9082149292705877e-10
    )
