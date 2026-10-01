"""CPU regularized lower incomplete gamma; scalar Float32/Float64 contract."""
from std.math import expm1, isfinite, isnan, lgamma
from numax.core.libm import exp, log


def _log1pmx(t: Float64) -> Float64:
    # log(1+t)-t without cancellation near x=a. The alternating power
    # series converges geometrically for |t| <= 1/2.
    var power = -t * t
    var total = power / 2
    for n in range(3, 80):
        power *= -t
        var term = power / Float64(n)
        total += term
        if abs(term) <= 1e-17 * abs(total):
            break
    return total


def _log_prefactor(a: Float64, x: Float64) -> Float64:
    # log(x**a * exp(-x) / Gamma(a)). Stirling's correction avoids
    # subtracting two O(a log(a)) numbers near the gamma distribution's mode.
    if a < 16:
        return a * log(x) - x - lgamma(a)
    var t = (x - a) / a
    var deviance: Float64
    if abs(t) <= 0.5:
        deviance = a * _log1pmx(t)
    else:
        var ratio = x / a
        var log_ratio = log(ratio) if ratio > 0 and isfinite(ratio) else log(x) - log(a)
        deviance = a * log_ratio + (a - x)
    var inv = 1 / a
    var inv2 = inv * inv
    var correction = inv * (1.0 / 12 + inv2 * (-1.0 / 360 + inv2 * (
        1.0 / 1260 + inv2 * (-1.0 / 1680 + inv2 * (1.0 / 1188
        - inv2 * 691.0 / 360360)))))
    return deviance + 0.5 * log(a) - 0.91893853320467274178 - correction


def _probability(value: Float64) raises -> Float64:
    if not isfinite(value) or value < -4e-14 or value > 1 + 4e-14:
        raise Error("ArithmeticError: gammainc numerical evaluation failed")
    return min(Float64(1), max(Float64(0), value))


def gammainc(a: Float64, x: Float64) raises -> Float64:
    """Return P(a,x) for finite a>0 and x>=0 (including positive infinity).

    Invalid parameters raise Error. Failure to converge within 100000 terms
    raises Error rather than returning a truncated approximation. Scalar CPU
    operation; no allocation, random state, Python or array broadcasting.
    """
    if not isfinite(a) or a <= 0 or isnan(x) or x < 0:
        raise Error("ValueError: gammainc requires finite a > 0 and x >= 0")
    if x == 0:
        return 0
    if not isfinite(x):
        return 1
    if a == 1:
        return -expm1(-x)

    var log_factor = _log_prefactor(a, x)
    # Far tails: the gamma Chernoff bound is exp(a*log(x/a)+a-x).
    # These thresholds are below half a subnormal / half an ulp at one.
    if a >= 16:
        var bound_log = log_factor - 0.5 * log(a) + 0.91893853320467274178
        # Dropping the positive Stirling correction makes this an estimate;
        # the margins (746 vs 745.14, 40 vs 37.43) exceed that correction.
        if x < a and bound_log < -746:
            return 0
        if x > a and bound_log < -40:
            return 1

    # Subtract instead of adding one to a: for large a, a+1 rounds to a.
    if x <= a or x - a < 1:
        # DLMF 8.7.1, normalized to begin at 1 (not 1/a).
        var term = Float64(1)
        var total = Float64(1)
        var compensation = Float64(0)
        for n in range(1, 100001):
            term *= x / (a + Float64(n))
            var corrected = term - compensation
            var next_total = total + corrected
            compensation = (next_total - total) - corrected
            total = next_total
            var ratio = x / (a + Float64(n + 1))
            # Bound the entire positive remainder by a geometric tail.
            if term * ratio <= 2e-16 * total * (1 - ratio):
                var scale = log_factor - log(a)
                if a < 16:
                    scale = a * log(x) - x - lgamma(a + 1)
                return _probability(exp(scale + log(total)))
    else:
        # Upper gamma continued fraction, evaluated by modified Lentz.
        # b_n=x+1-a+2n, numerator n*(a-n). This branch has b_0>=2.
        var b = (x - a) + 1
        var c = Float64(1e300)
        var d = 1 / b
        var h = d
        for n in range(1, 100001):
            var an = Float64(n) * (a - Float64(n))
            b += 2
            d = b + an * d
            c = b + an / c
            if abs(d) < 1e-300:
                d = 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1 / d
            var delta = c * d
            h *= delta
            if abs(delta - 1) <= 4e-16:
                return _probability(-expm1(log_factor + log(h)))
            if not isfinite(h):
                raise Error("ArithmeticError: gammainc continued fraction overflow")
    raise Error("ArithmeticError: gammainc did not converge within 100000 terms")


def gammainc(a: Float32, x: Float32) raises -> Float32:
    """Widen Float32 arguments, evaluate in Float64, round once on return."""
    return Float32(gammainc(Float64(a), Float64(x)))
