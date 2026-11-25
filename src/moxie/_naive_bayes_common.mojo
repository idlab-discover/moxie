"""Shared Naive Bayes validation; formulas retain sklearn attribution."""
from std.math import isfinite
import numojo as nm
from numax.core.libm import exp as _libm_exp, log as _libm_log
from moxie.base import IncrementalClassifier, LogProbabilityPredictor
from moxie.warnings import FitWarning

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
    counts: nm.NDArray[DType.float64],
    prior: Optional[List[Float64]],
    fit_prior: Bool,
) raises -> nm.NDArray[DType.float64]:
    var n_classes = counts.size
    _check_prior(prior, n_classes)
    var result = nm.zeros[DType.float64](nm.Shape(n_classes))
    if prior:
        for c in range(n_classes):
            result[nm.Item(c)] = _log(prior.value()[c])
    elif fit_prior:
        # Preserve the ordered Float64 sum used for the empirical priors.
        var total = Float64(0)
        for c in range(n_classes):
            total += counts.item(c)
        var log_total = _log(total)
        for c in range(n_classes):
            result[nm.Item(c)] = _log(counts.item(c)) - log_total
    else:
        var log_uniform = -_log(Float64(n_classes))
        for c in range(n_classes):
            result[nm.Item(c)] = log_uniform
    return result^


comptime _ALPHA_FLOOR: Float64 = 1e-10


def _alpha(alpha: Float64, force_alpha: Bool) raises -> Float64:
    if not isfinite(alpha) or alpha < 0:
        raise Error("ValueError: alpha must be finite and nonnegative")
    if not force_alpha and alpha < _ALPHA_FLOOR:
        return _ALPHA_FLOOR
    return alpha


def _alpha_fit_warnings(alpha: Float64, force_alpha: Bool) -> List[FitWarning]:
    if not force_alpha and alpha < _ALPHA_FLOOR:
        return [FitWarning(
            "UserWarning",
            "alpha too small will result in numeric errors, setting alpha = "
            "1.0e-10. Use `force_alpha=True` to keep alpha unchanged.",
        )]
    return []


def _check_binarize(threshold: Optional[Float64]) raises:
    if threshold:
        if not isfinite(threshold.value()) or threshold.value() < 0:
            raise Error(
                "ValueError: binarize must be None or finite and nonnegative"
            )


