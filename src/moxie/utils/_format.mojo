# Moved from the Moxie tree exporter adapted from sklearn/tree/_export.py.
# Copyright the scikit-learn developers; see LICENSES/scikit-learn.txt.
"""Internal text formatting shared by Moxie estimators."""
from std.memory import bitcast


def format_fixed(value: Float64, decimals: Int) raises -> String:
    """Exact binary64 to fixed decimal, rounded to nearest with ties to even.

    String.format lacks precision support on Mojo 1.0; NuMojo's formatter
    scales through floats and Int, losing digits and overflowing large values.
    """
    if decimals < 0:
        raise Error("ValueError: decimals must be nonnegative")
    var bits = bitcast[DType.uint64](value)
    var negative = (bits >> 63) != 0
    var exponent = Int((bits >> 52) & 2047)
    var mantissa = Int(bits & 0x000fffffffffffff)
    if exponent == 2047:
        if mantissa != 0:
            return "nan"
        return "-inf" if negative else "inf"
    if exponent != 0:
        mantissa += 4503599627370496
    var power = (exponent - 1023 - 52) if exponent != 0 else -1074
    var digits = List[Int]()
    while mantissa != 0:
        digits.append(mantissa % 10)
        mantissa //= 10
    if len(digits) == 0:
        digits.append(0)
    var scale = max(0, -power)
    for _ in range(abs(power)):
        var carry = 0
        for i in range(len(digits)):
            var product = digits[i] * (2 if power >= 0 else 5) + carry
            digits[i] = product % 10
            carry = product // 10
        if carry != 0:
            digits.append(carry)
    while len(digits) <= scale:
        digits.append(0)
    if decimals < scale:
        var cut = scale - decimals
        var sticky = False
        for i in range(cut - 1):
            sticky = sticky or digits[i] != 0
        var up = digits[cut - 1] > 5 or (digits[cut - 1] == 5 and (sticky or digits[cut] % 2 != 0))
        var rounded = List[Int]()
        var carry = Int(up)
        for i in range(cut, len(digits)):
            var digit = digits[i] + carry
            rounded.append(digit % 10)
            carry = digit // 10
        if carry:
            rounded.append(carry)
        digits = rounded^
        scale = decimals
    var result = String("-" if negative else "")
    for i in range(len(digits) - 1, scale - 1, -1):
        result += String(digits[i])
    if decimals > 0:
        result += "."
        for i in range(scale - 1, -1, -1):
            result += String(digits[i])
        for _ in range(decimals - scale):
            result += "0"
    return result
