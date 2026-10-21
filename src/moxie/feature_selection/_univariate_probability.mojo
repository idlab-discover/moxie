"""Scalar F and chi-square survival functions; DLMF 8.17 and 8.9."""
from std.math import lgamma, isnan, isinf, nan
# std.math log/exp/log1p are only ~1e-10 (log1p ~1e-7) accurate for Float64;
# deep tails multiply the log by df/2, so use Numax's fdlibm transcriptions.
# Callers pass x and its complement y = 1 - x separately so neither is formed
# by cancellation.
from numax.core.libm import exp, log

def _nonzero(x: Float64) -> Float64:
    if abs(x) < 1e-300:
        return -1e-300 if x < 0 else 1e-300
    return x

def _beta_fraction(a: Float64, b: Float64, x: Float64) raises -> Float64:
    var c = Float64(1)
    var d = 1 / _nonzero(1 - (a + b) * x / (a + 1))
    var h = d
    for index in range(1, 10001):
        var m = Float64(index)
        var twice = 2 * m
        var coefficient = m * (b - m) * x / ((a + twice - 1) * (a + twice))
        d = 1 / _nonzero(1 + coefficient * d)
        c = _nonzero(1 + coefficient / c)
        h *= d * c
        coefficient = -(a + m) * (a + b + m) * x / ((a + twice) * (a + twice + 1))
        d = 1 / _nonzero(1 + coefficient * d)
        c = _nonzero(1 + coefficient / c)
        var delta = d * c
        h *= delta
        if abs(delta - 1) < 4e-15:
            return h
    raise Error("ArithmeticError: incomplete beta failed to converge")

def _beta(a: Float64, b: Float64, x: Float64, y: Float64) raises -> Float64:
    """Regularized I_x(a, b) with y = 1 - x supplied accurately."""
    if x <= 0:
        return 0
    if y <= 0:
        return 1
    var factor = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(y))
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_fraction(a, b, x) / a
    return 1 - factor * _beta_fraction(b, a, y) / b

def f_survival(f: Float64, df1: Float64, df2: Float64) raises -> Float64:
    if isnan(f) or df1 <= 0 or df2 <= 0:
        return nan[DType.float64]()
    if f < 0:
        return 1
    if isinf(f):
        return 0
    var t = df1 * f / df2
    return _beta(df2 / 2, df1 / 2, 1 / (1 + t), t / (1 + t))

def chi_survival(statistic: Float64, degrees: Float64) raises -> Float64:
    if isnan(statistic) or statistic < 0 or degrees <= 0:
        return nan[DType.float64]()
    if statistic == 0:
        return 1
    if isinf(statistic):
        return 0
    var a = degrees / 2
    var x = statistic / 2
    var factor = exp(a * log(x) - x - lgamma(a))
    if x < a + 1:
        var term = 1 / a
        var total = term
        for index in range(1, 10001):
            term *= x / (a + Float64(index))
            total += term
            if abs(term) < abs(total) * 4e-15:
                return max(Float64(0), 1 - total * factor)
    else:
        var b = x + 1 - a
        var c = Float64(1e300)
        var d = 1 / _nonzero(b)
        var h = d
        for index in range(1, 10001):
            var i = Float64(index)
            var coefficient = -i * (i - a)
            b += 2
            d = 1 / _nonzero(coefficient * d + b)
            c = _nonzero(b + coefficient / c)
            var delta = d * c
            h *= delta
            if abs(delta - 1) < 4e-15:
                return h * factor
    raise Error("ArithmeticError: incomplete gamma failed to converge")
