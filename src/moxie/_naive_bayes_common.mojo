"""Shared Naive Bayes validation; formulas retain sklearn attribution."""
from std.math import isfinite
from numax.core.libm import exp as _libm_exp, log as _libm_log
from moxie.base import IncrementalClassifier, LogProbabilityPredictor

trait NaiveBayesClassifier(IncrementalClassifier, LogProbabilityPredictor):
    pass


comptime _Matrix = List[List[Float64]]


def _exp(value: Float64) -> Float64:
    # Mojo 1.0's std.math exp/log are not double accurate (up to ~2e7 ulp for
    # log); a log ulp can flip a sklearn tie. Numax's fdlibm transcriptions are
    # within one ulp of glibc, return NaN for NaN and handle +/-inf like libm.
    return _libm_exp(value)


def _log(value: Float64) -> Float64:
    return _libm_log(value)


def _sum(values: List[Float64]) -> Float64:
    var result = Float64(0)
    for value in values:
        result += value
    return result


def _check_X(
    X: _Matrix, expected: Int = 0, nonnegative: Bool = False
) raises -> Int:
    if len(X) == 0 or len(X[0]) == 0:
        raise Error("ValueError: X must contain samples and features")
    var n = len(X[0])
    if expected != 0 and n != expected:
        raise Error("ValueError: feature count differs from fit")
    for row in X:
        if len(row) != n:
            raise Error("ValueError: X must be rectangular")
        for value in row:
            if not isfinite(value):
                raise Error("ValueError: X must be finite")
            if nonnegative and value < 0:
                raise Error("ValueError: X must be nonnegative")
    return n


def _weights(
    n: Int, sample_weight: Optional[List[Float64]]
) raises -> List[Float64]:
    var result = List[Float64]()
    if sample_weight:
        result = sample_weight.value().copy()
        if len(result) != n:
            raise Error("ValueError: sample_weight length differs from X")
    else:
        for _ in range(n):
            result.append(1.0)
    for value in result:
        if not isfinite(value) or value < 0:
            raise Error("ValueError: weights must be finite and nonnegative")
    if not isfinite(_sum(result)) or _sum(result) <= 0:
        raise Error(
            "ValueError: total sample weight must be positive and finite"
        )
    return result^


def _scalar_weights(n: Int, sample_weight: Float64) raises -> List[Float64]:
    """Broadcast a scalar; the existing batch validator checks the total."""
    if not isfinite(sample_weight) or sample_weight <= 0:
        raise Error("ValueError: scalar sample weight must be positive and finite")
    return List[Float64](length=n, fill=sample_weight)


def _check_prior(
    prior: Optional[List[Float64]], n: Int, normalized: Bool = False
) raises:
    if not prior:
        return
    if len(prior.value()) != n:
        raise Error("ValueError: number of priors must match number of classes")
    for value in prior.value():
        if not isfinite(value) or value < 0:
            raise Error("ValueError: priors must be finite and nonnegative")
    if normalized and abs(_sum(prior.value()) - 1.0) > 1.001e-5:
        raise Error("ValueError: Gaussian priors must sum to one")


def _log_priors(
    counts: List[Float64], prior: Optional[List[Float64]], fit_prior: Bool
) raises -> List[Float64]:
    _check_prior(prior, len(counts))
    var result = List[Float64]()
    for i in range(len(counts)):
        if prior:
            result.append(_log(prior.value()[i]))
        elif fit_prior:
            result.append(_log(counts[i]) - _log(_sum(counts)))
        else:
            result.append(-_log(Float64(len(counts))))
    return result^


def _alpha(alpha: Float64, force_alpha: Bool) raises -> Float64:
    if not isfinite(alpha) or alpha < 0:
        raise Error("ValueError: alpha must be finite and nonnegative")
    if not force_alpha and alpha < 1e-10:
        return 1e-10
    return alpha


def _check_binarize(threshold: Optional[Float64]) raises:
    if threshold:
        if not isfinite(threshold.value()) or threshold.value() < 0:
            raise Error(
                "ValueError: binarize must be None or finite and nonnegative"
            )


