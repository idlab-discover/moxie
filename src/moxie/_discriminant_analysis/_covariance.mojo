# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Covariance adapters shared by LDA and QDA (sklearn `_cov`, `_class_means`, `_class_cov`).

Reuses Moxie covariance kernels on StandardScaler-standardized data,
as in `discriminant_analysis._cov`.
All arithmetic uses Float64 work arrays.
"""
from moxie.covariance import empirical_covariance, shrunk_covariance
from moxie.covariance._shrunk_covariance_adaptive import (
    ledoit_wolf as covariance_ledoit_wolf, ledoit_wolf_shrinkage,
)
from moxie.covariance._empirical_covariance import (
    column_means,
)
from std.math import sqrt
from numojo import NDArray
from ._work import Work, get, put, matrix, vector

comptime EPS64: Float64 = 2.220446049250313e-16


struct Shrinkage(Copyable, Movable, Writable):
    """The `shrinkage` parameter: None, the string 'auto' or a fixed float in [0, 1].

    Implicit constructors let callers write `shrinkage=None`, `shrinkage="auto"`
    or `shrinkage=0.5`. Values are validated at fit time, like sklearn.
    """

    var text: Optional[String]
    var value: Optional[Float64]

    def __init__(out self):
        """No shrinkage (`shrinkage=None`)."""
        self.text = None
        self.value = None

    @implicit
    def __init__(out self, none: NoneType):
        self.text = None
        self.value = None

    @doc_hidden
    @implicit
    def __init__(out self, none: NoneType._mlir_type):
        self.text = None
        self.value = None

    @implicit
    def __init__(out self, text: String):
        self.text = text
        self.value = None

    @implicit
    def __init__(out self, text: StringLiteral):
        self.text = String(text)
        self.value = None

    @implicit
    def __init__(out self, value: Float64):
        self.text = None
        self.value = value

    @implicit
    def __init__(out self, value: Int):
        self.text = None
        self.value = Float64(value)

    def is_none(self) -> Bool:
        """True for `shrinkage=None` (no shrinkage)."""
        return not self.text and not self.value

    def is_auto(self) -> Bool:
        """True when the Ledoit-Wolf 'auto' option was requested."""
        return Bool(self.text)

    def is_zero(self) -> Bool:
        """True for the numeric value 0 (allowed together with a covariance estimator)."""
        return Bool(self.value) and self.value.value() == 0

    def validate(self, estimator: String) raises:
        """Raise ValueError unless this is None, 'auto' or a float in [0, 1]."""
        if self.text:
            if self.text.value() != "auto":
                raise Error(
                    String("ValueError: The 'shrinkage' parameter of ", estimator)
                    + String(" must be a str among {'auto'}, a float in the range [0, 1] or None. Got '")
                    + self.text.value()
                    + "' instead."
                )
        elif self.value:
            var v = self.value.value()
            if not (v >= 0 and v <= 1):
                raise Error(
                    String("ValueError: The 'shrinkage' parameter of ", estimator)
                    + String(" must be a str among {'auto'}, a float in the range [0, 1] or None. Got ")
                    + String(v)
                    + " instead."
                )

    def write_to(self, mut writer: Some[Writer]):
        if self.text:
            writer.write("'", self.text.value(), "'")
        elif self.value:
            writer.write(self.value.value())
        else:
            writer.write("None")


trait CovarianceEstimator(Copyable, Movable, Deinitable):
    """Native adaptation of sklearn's `covariance_estimator` duck type.

    sklearn requires an object with `fit(X)` and a `covariance_` attribute.
    Here `fit` receives a compact Float64 (n_samples, n_features) copy of the
    (per-class) samples and `covariance` returns the fitted (p, p) matrix.
    Estimators keep a private copy: the configured object is never mutated.
    """

    def fit(mut self, X: NDArray[DType.float64]) raises:
        """Estimate the covariance of the rows of X."""
        ...

    def covariance(self) raises -> NDArray[DType.float64]:
        """Return the fitted (n_features, n_features) covariance matrix."""
        ...


@fieldwise_init
struct NoCovarianceEstimator(CovarianceEstimator, ImplicitlyCopyable):
    """Type placeholder for `covariance_estimator=None`; never fitted."""

    def fit(mut self, X: NDArray[DType.float64]) raises:
        raise Error("ValueError: NoCovarianceEstimator is only a type placeholder for None")

    def covariance(self) raises -> NDArray[DType.float64]:
        raise Error("ValueError: NoCovarianceEstimator is only a type placeholder for None")


def ledoit_wolf(X: Work) raises -> Work:
    """Covariance component of the shared Ledoit-Wolf result."""
    var result = covariance_ledoit_wolf(X)
    return result[0].copy()


def ledoit_wolf_shrinkage_centered(Xc: Work) raises -> Float64:
    """Shared coefficient for already-centered Float64 samples."""
    return ledoit_wolf_shrinkage(Xc, assume_centered=True)


def standard_scale(X: Work) raises -> Work:
    """Per-feature StandardScaler scale_ (population std; near-constant features -> 1).

    Mean/variance follow sklearn's `_incremental_mean_and_var` first pass (with its
    correction term) and `_is_constant_feature` / `_handle_zeros_in_scale`.
    """
    var n = X.shape[0]
    var p = X.shape[1]
    var nf = Float64(n)
    var means = column_means(X)
    var scale = vector(p)
    for j in range(p):
        var mean = means.unsafe_get(j)
        var correction: Float64 = 0
        var squares: Float64 = 0
        for s in range(n):
            var d = X.unsafe_get(s * p + j) - mean
            correction += d
            squares += d * d
        var variance = (squares - correction * correction / nf) / nf
        var bound = nf * EPS64 * variance + (nf * mean * EPS64) ** 2
        scale.unsafe_set(j, 1.0 if variance <= bound else sqrt(variance))
    return scale^


def standardized_ledoit_wolf(X: Work) raises -> Work:
    """Equivalent of `_cov(X, 'auto')`: Ledoit-Wolf on standardized X, rescaled by scale_."""
    var n = X.shape[0]
    var p = X.shape[1]
    var means = column_means(X)
    var scale = standard_scale(X)
    var Xs = matrix(n, p)
    for s in range(n):
        for j in range(p):
            Xs.unsafe_set(s * p + j, (X.unsafe_get(s * p + j) - means.unsafe_get(j)) / scale.unsafe_get(j))
    var c = ledoit_wolf(Xs)
    for i in range(p):
        for j in range(p):
            put(c, i, j, scale.unsafe_get(i) * get(c, i, j) * scale.unsafe_get(j))
    return c^


def estimate_covariance[
    C: CovarianceEstimator
 ](X: Work, shrinkage: Shrinkage, estimator: Optional[C]) raises -> Work:
    """Equivalent of `_cov`: empirical, Ledoit-Wolf ('auto'), fixed shrinkage or custom estimator."""
    if estimator:
        if not shrinkage.is_none() and not shrinkage.is_zero():
            raise Error(
                "ValueError: covariance_estimator and shrinkage parameters are not None."
                " Only one of the two can be set."
            )
        var fitted = estimator.value().copy()
        fitted.fit(X.copy())
        var c = fitted.covariance()
        var p = X.shape[1]
        if c.ndim != 2 or c.shape[0] != p or c.shape[1] != p:
            raise Error(
                String("ValueError: covariance_estimator returned a covariance of shape ")
                + String(c.shape)
                + String(", expected (", p, ", ", p, ")")
            )
        var out = matrix(p, p)
        var s0 = Int(c.strides[0])
        var s1 = Int(c.strides[1])
        for i in range(p):
            for j in range(p):
                put(out, i, j, c.unsafe_get(i * s0 + j * s1))
        return out^
    if shrinkage.is_none():
        return empirical_covariance(X)
    if shrinkage.is_auto():
        return standardized_ledoit_wolf(X)
    return shrunk_covariance(empirical_covariance(X), shrinkage.value.value())


def class_rows(X: Work, codes: List[Int], target: Int, count: Int) raises -> Work:
    """Rows of X whose code equals `target`, in sample order."""
    var p = X.shape[1]
    var out = matrix(count, p)
    var r = 0
    for s in range(X.shape[0]):
        if codes[s] == target:
            for j in range(p):
                out.unsafe_set(r * p + j, X.unsafe_get(s * p + j))
            r += 1
    debug_assert(r == count, "class count mismatch")
    return out^


def class_means(X: Work, codes: List[Int], counts: List[Int]) raises -> Work:
    """Per-class means, shape (n_classes, p) (sklearn `_class_means`)."""
    var k = len(counts)
    var p = X.shape[1]
    var means = matrix(k, p)
    for s in range(X.shape[0]):
        var c = codes[s]
        for j in range(p):
            means.unsafe_set(c * p + j, means.unsafe_get(c * p + j) + X.unsafe_get(s * p + j))
    for c in range(k):
        for j in range(p):
            put(means, c, j, get(means, c, j) / Float64(counts[c]))
    return means^


def class_covariance[
    C: CovarianceEstimator
](
    X: Work,
    codes: List[Int],
    counts: List[Int],
    priors: Work,
    shrinkage: Shrinkage,
    estimator: Optional[C],
) raises -> Work:
    """Prior-weighted within-class covariance sum_k prior_k * _cov(X_k) (sklearn `_class_cov`)."""
    var p = X.shape[1]
    var total = matrix(p, p)
    for k in range(len(counts)):
        var ck = estimate_covariance(class_rows(X, codes, k, counts[k]), shrinkage, estimator)
        var w = priors.unsafe_get(k)
        for i in range(total.size):
            total.unsafe_set(i, total.unsafe_get(i) + w * ck.unsafe_get(i))
    return total^

def centered(X: Work, means: Work) raises -> Work:
    """Return X - means (row broadcast)."""
    var n = X.shape[0]
    var p = X.shape[1]
    var out = matrix(n, p)
    for i in range(n):
        for j in range(p):
            out.unsafe_set(i * p + j, X.unsafe_get(i * p + j) - means.unsafe_get(j))
    return out^
