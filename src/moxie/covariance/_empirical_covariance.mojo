# Adapted from scikit-learn sklearn/covariance/_empirical_covariance.py and
# sklearn/utils/extmath.py (fast_logdet); authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Maximum-likelihood covariance, precision, likelihood and Mahalanobis kernels.

This module owns the family's shared dense numerics (`CovarianceModel`,
`EmpiricalCovarianceEngine` and the free helpers) and the public
`EmpiricalCovariance`, `empirical_covariance` and `log_likelihood`.
All reductions accumulate in Float64; array results keep the input dtype.
"""
import numojo as nm
from numojo import NDArray, Shape
from std.math import sqrt, isfinite, abs
from std.sys import llvm_intrinsic, stderr
from mojo_linalg import symmetric_pinv, symmetric_slogdet, symmetric_eigh, qr
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._engines import CenteredData, CovarianceEngine
from moxie.covariance._state import CovarianceState, validate_matrix

comptime _LOG_TWO_PI: Float64 = 1.8378770664093453
comptime _SIMD_WIDTH = 8


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


@always_inline
def libm_log(x: Float64) -> Float64:
    """Natural logarithm through the platform libm (std.math.log is less accurate)."""
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


def machine_epsilon[dtype: DType]() -> Float64:
    """Unit roundoff spacing at 1.0 (NumPy `finfo(dtype).eps`)."""
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-07
    return 2.220446049250313e-16


def check_float_dtype[dtype: DType]():
    comptime assert dtype == DType.float32 or dtype == DType.float64, "covariance supports Float32 or Float64"


@always_inline
def at[dtype: DType](x: NDArray[dtype], i: Int, j: Int, row_stride: Int, col_stride: Int) -> Float64:
    """Element (i, j) given precomputed element strides (NuMojo applies the offset)."""
    return Float64(x.unsafe_get(i * row_stride + j * col_stride))


def entry[dtype: DType](x: NDArray[dtype], i: Int, j: Int) raises -> Float64:
    """Logical element (i, j) of a rank-2 array, honoring strides and offset."""
    return Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))


def shape_text[dtype: DType](x: NDArray[dtype]) raises -> String:
    var text = String("(")
    for k in range(x.ndim):
        if k:
            text += ", "
        text += String(x.shape[k])
    if x.ndim == 1:
        text += ","
    return text + ")"


def validate_samples[dtype: DType](x: NDArray[dtype], owner: String, min_samples: Int = 1, min_features: Int = 1) raises:
    """Sklearn `check_array` for estimator input: 2-D, finite, minimum sizes.

    Reuses the family validator and adds the estimator/shape context.
    """
    check_float_dtype[dtype]()
    if x.ndim != 2:
        raise Error(String("ValueError: ", owner, ": expected a 2-D array, got shape ", shape_text(x)))
    if x.shape[0] < min_samples:
        raise Error(String("ValueError: ", owner, ": found array with ", x.shape[0], " sample(s) (shape=",
                           shape_text(x), ") while a minimum of ", min_samples, " is required"))
    if x.shape[1] < min_features:
        raise Error(String("ValueError: ", owner, ": found array with ", x.shape[1], " feature(s) (shape=",
                           shape_text(x), ") while a minimum of ", min_features, " is required"))
    try:
        validate_matrix(x, min_samples)
    except e:
        raise Error(String("ValueError: ", owner, ": input contains NaN or infinity (shape=", shape_text(x), "): ", e))


def to_array[dtype: DType](values: NDArray[DType.float64]) raises -> NDArray[dtype]:
    """Cast a contiguous Float64 work array into a new array of `dtype`."""
    var shape = List[Int]()
    for k in range(values.ndim):
        shape.append(values.shape[k])
    var result = nm.zeros[dtype](Shape(shape))
    for k in range(values.size):
        result.unsafe_set(k, values.unsafe_get(k).cast[dtype]())
    return result^


def to_float64[dtype: DType](x: NDArray[dtype]) raises -> NDArray[DType.float64]:
    """Contiguous Float64 copy of a rank-2 array (any valid strides)."""
    var rows = x.shape[0]
    var cols = x.shape[1]
    var rs = Int(x.strides[0])
    var cs = Int(x.strides[1])
    var result = nm.zeros[DType.float64](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            result.unsafe_set(i * cols + j, at(x, i, j, rs, cs))
    return result^


# ---------------------------------------------------------------------------
# Centering and empirical covariance
# ---------------------------------------------------------------------------


def column_means[dtype: DType](x: NDArray[dtype]) raises -> NDArray[DType.float64]:
    """Float64 column means of an (n, p) array."""
    var n = x.shape[0]
    var p = x.shape[1]
    var rs = Int(x.strides[0])
    var cs = Int(x.strides[1])
    var result = nm.zeros[DType.float64](Shape(p))
    for j in range(p):
        var total: Float64 = 0
        for i in range(n):
            total += at(x, i, j, rs, cs)
        result.unsafe_set(j, total / Float64(n))
    return result^


def centered_columns[dtype: DType](x: NDArray[dtype], location: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """Column-major Float64 copy (p, n) of `x - location`, contiguous per feature."""
    var n = x.shape[0]
    var p = x.shape[1]
    var rs = Int(x.strides[0])
    var cs = Int(x.strides[1])
    var columns = nm.zeros[DType.float64](Shape(p, n))
    for j in range(p):
        var shift = location.unsafe_get(j)
        for i in range(n):
            columns.unsafe_set(j * n + i, at(x, i, j, rs, cs) - shift)
    return columns^


@always_inline
def column_dot(columns: NDArray[DType.float64], a: Int, b: Int, n: Int) -> Float64:
    """Dot product of rows `a` and `b` of a contiguous (p, n) work array."""
    var accumulator = SIMD[DType.float64, _SIMD_WIDTH](0)
    var first = a * n
    var second = b * n
    var k = 0
    while k + _SIMD_WIDTH <= n:
        accumulator += columns.unsafe_load[width=_SIMD_WIDTH](first + k) * columns.unsafe_load[width=_SIMD_WIDTH](second + k)
        k += _SIMD_WIDTH
    var total = accumulator.reduce_add()
    while k < n:
        total += columns.unsafe_get(first + k) * columns.unsafe_get(second + k)
        k += 1
    return total


def gram(columns: NDArray[DType.float64], divisor: Float64) raises -> NDArray[DType.float64]:
    """Symmetric (p, p) matrix C Cᵀ / divisor for column-major data C (p, n)."""
    var p = columns.shape[0]
    var n = columns.shape[1]
    var result = nm.zeros[DType.float64](Shape(p, p))
    for a in range(p):
        for b in range(a + 1):
            var value = column_dot(columns, a, b, n) / divisor
            result.unsafe_set(a * p + b, value)
            result.unsafe_set(b * p + a, value)
    return result^


def empirical_covariance_float64[dtype: DType](x: NDArray[dtype], assume_centered: Bool) raises -> NDArray[DType.float64]:
    """Float64 maximum-likelihood covariance (divisor n) of a rank-2 array."""
    var p = x.shape[1]
    if x.shape[0] == 1:
        print("UserWarning: Only one sample available. You may want to reshape your data array", file=stderr)
    var location: NDArray[DType.float64]
    if assume_centered:
        location = nm.zeros[DType.float64](Shape(p))
    else:
        location = column_means(x)
    return gram(centered_columns(x, location), Float64(x.shape[0]))


def empirical_covariance[dtype: DType](X: NDArray[dtype], *, assume_centered: Bool = False) raises -> NDArray[dtype]:
    """Maximum likelihood covariance estimator (sklearn `empirical_covariance`).

    X is (n_samples, n_features), or a 1-D array read as a single sample.
    Returns the (n_features, n_features) covariance with divisor n_samples
    (`np.cov(X.T, bias=1)`), centered unless `assume_centered`. Like sklearn,
    nonfinite entries are not rejected and propagate. Accumulation is Float64;
    the result keeps the input dtype (sklearn returns Float64 when centering).
    """
    check_float_dtype[dtype]()
    if X.ndim == 1:
        if X.shape[0] < 1:
            raise Error("ValueError: empirical_covariance: found array with 0 feature(s) (shape=(1, 0))")
        var row = nm.zeros[dtype](Shape(1, X.shape[0]))
        for j in range(X.shape[0]):
            row.unsafe_set(j, X.unsafe_get(j * Int(X.strides[0])))
        return to_array[dtype](empirical_covariance_float64(row, assume_centered))
    if X.ndim != 2:
        raise Error(String("ValueError: empirical_covariance: expected a 1-D or 2-D array, got shape ", shape_text(X)))
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise Error(String("ValueError: empirical_covariance: found empty array with shape ", shape_text(X)))
    return to_array[dtype](empirical_covariance_float64(X, assume_centered))


def center[dtype: DType](x: NDArray[dtype], assume_centered: Bool) raises -> CenteredData[dtype]:
    """Location and independently owned centered samples (family engine contract)."""
    var n = x.shape[0]
    var p = x.shape[1]
    var location: NDArray[DType.float64]
    if assume_centered:
        location = nm.zeros[DType.float64](Shape(p))
    else:
        location = column_means(x)
    var rs = Int(x.strides[0])
    var cs = Int(x.strides[1])
    var values = nm.zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p):
            values.unsafe_set(i * p + j, (at(x, i, j, rs, cs) - location.unsafe_get(j)).cast[dtype]())
    return CenteredData[dtype](values^, to_array[dtype](location))


# ---------------------------------------------------------------------------
# Precision, determinants, likelihood, distances and error norms
# ---------------------------------------------------------------------------


def pinvh[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """Symmetric pseudoinverse with scipy `pinvh` defaults.

    Eigenvalues with |w| <= n * eps(dtype) * max|w| are discarded (atol=0);
    the lower triangle is read. Zero and rank-deficient matrices are valid.
    """
    var n = a.shape[0]
    try:
        var result = symmetric_pinv(a, rtol=Float64(n) * machine_epsilon[dtype](), atol=0.0)
        return result.matrix.copy()
    except e:
        raise Error(String("LinAlgError: symmetric pseudoinverse failed (", e.kind, " in ", e.operation, ", index ", e.index,
                           ") for a ", n, "x", n, " matrix"))


def fast_logdet[dtype: DType](a: NDArray[dtype]) raises -> Float64:
    """Return log(det(A)) when the determinant is positive, otherwise -inf (sklearn `fast_logdet`).

    Symmetric inputs use LDLᵀ. Nonsymmetric inputs use pivoted Householder
    QR, including the column permutation sign; no determinant product is formed.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise Error("ValueError: fast_logdet: expected a square matrix")
    var n = a.shape[0]
    var symmetric = True
    for i in range(n):
        for j in range(n):
            if not isfinite(entry(a, i, j)):
                raise Error("ValueError: fast_logdet: array must not contain infs or NaNs")
            if entry(a, i, j) != entry(a, j, i):
                symmetric = False
    try:
        if not symmetric:
            # A[:, permutation] = Q R. Each nonzero Householder reflector
            # contributes -1. A zero R diagonal makes the determinant zero.
            var factor = qr(to_float64(a))
            var r = factor.r()
            var permutation = factor.permutation()
            var sign = 1 if n % 2 == 0 else -1
            var logabsdet: Float64 = 0
            for i in range(n):
                var diagonal = r.unsafe_get(i * n + i)
                if diagonal == 0:
                    return -Float64.MAX * 2.0
                if diagonal < 0:
                    sign = -sign
                logabsdet += libm_log(abs(diagonal))
                for j in range(i):
                    if permutation[j] > permutation[i]:
                        sign = -sign
            return logabsdet if sign > 0 else -Float64.MAX * 2.0
        var result = symmetric_slogdet(a)
        if result.sign <= 0:
            return -Float64.MAX * 2.0
        return result.logabsdet
    except e:
        raise Error(String("LinAlgError: signed log determinant failed (", e, ")"))


def log_likelihood[dtype: DType, precision_dtype: DType = dtype](emp_cov: NDArray[dtype], precision: NDArray[precision_dtype]) raises -> Float64:
    """Sample mean Gaussian log-likelihood (sklearn `log_likelihood`).

    Returns (-sum(emp_cov * precision) + fast_logdet(precision) - p log(2 pi)) / 2.
    Rank-2 operands broadcast as in sklearn; precision must remain square.
    """
    check_float_dtype[dtype]()
    if emp_cov.ndim != 2 or precision.ndim != 2 or precision.shape[0] != precision.shape[1] \
            or (emp_cov.shape[0] != precision.shape[0] and emp_cov.shape[0] != 1 and precision.shape[0] != 1) \
            or (emp_cov.shape[1] != precision.shape[1] and emp_cov.shape[1] != 1 and precision.shape[1] != 1):
        raise Error(String("ValueError: log_likelihood: emp_cov ", shape_text(emp_cov), " and precision ",
                           shape_text(precision), " must be broadcastable rank-2 arrays with square precision"))
    var p = precision.shape[0]
    var rows = p if emp_cov.shape[0] == 1 else emp_cov.shape[0]
    var columns = p if emp_cov.shape[1] == 1 else emp_cov.shape[1]
    var total: Float64 = 0
    for i in range(rows):
        for j in range(columns):
            total += entry(emp_cov, 0 if emp_cov.shape[0] == 1 else i, 0 if emp_cov.shape[1] == 1 else j) \
                * entry(precision, 0 if p == 1 else i, 0 if p == 1 else j)
    var result = -total + fast_logdet(precision)
    result -= Float64(p) * _LOG_TWO_PI
    return result / 2.0


def squared_mahalanobis[dtype: DType, precision_dtype: DType = dtype](x: NDArray[dtype], location: NDArray[dtype], precision: NDArray[precision_dtype]) raises -> NDArray[dtype]:
    """Per-row quadratic form, shape (n,); negative forms give NaN like sklearn."""
    var n = x.shape[0]
    var p = x.shape[1]
    var p64 = to_float64(precision)
    var rs = Int(x.strides[0])
    var cs = Int(x.strides[1])
    var ls = Int(location.strides[0])
    var difference = nm.zeros[DType.float64](Shape(p))
    var result = nm.zeros[dtype](Shape(n))
    for i in range(n):
        for j in range(p):
            difference.unsafe_set(j, at(x, i, j, rs, cs) - Float64(location.unsafe_get(j * ls)))
        var total: Float64 = 0
        for a in range(p):
            var row: Float64 = 0
            for b in range(p):
                row += p64.unsafe_get(a * p + b) * difference.unsafe_get(b)
            total += row * difference.unsafe_get(a)
        # sklearn computes scipy Mahalanobis distance and then squares it.
        # An indefinite precision can produce a negative form after a failed
        # graphical fit; preserve sqrt(negative) -> NaN without rounding valid forms.
        if total < 0:
            total = Float64("nan")
        result.unsafe_set(i, total.cast[dtype]())
    return result^


def error_norm_value[dtype: DType](covariance: NDArray[dtype], comparison: NDArray[dtype], norm: String = "frobenius",
                                   scaling: Bool = True, squared: Bool = True) raises -> Float64:
    """Squared Frobenius or spectral norm of `comparison - covariance` (sklearn `error_norm`)."""
    if comparison.ndim != 2 or comparison.shape[0] != covariance.shape[0] or comparison.shape[1] != covariance.shape[1]:
        raise Error(String("ValueError: error_norm: comp_cov shape ", shape_text(comparison), " differs from covariance_ shape ",
                           shape_text(covariance)))
    var p = covariance.shape[0]
    var squared_norm: Float64
    if norm == "frobenius":
        squared_norm = 0
        for i in range(p):
            for j in range(p):
                var d = entry(comparison, i, j) - entry(covariance, i, j)
                squared_norm += d * d
    elif norm == "spectral":
        # Largest singular value of the PSD matrix EᵀE equals its largest eigenvalue.
        var product = nm.zeros[DType.float64](Shape(p, p))
        for a in range(p):
            for b in range(a + 1):
                var value: Float64 = 0
                for k in range(p):
                    value += (entry(comparison, k, a) - entry(covariance, k, a)) * (entry(comparison, k, b) - entry(covariance, k, b))
                product.unsafe_set(a * p + b, value)
                product.unsafe_set(b * p + a, value)
        var largest: Float64 = 0
        try:
            var values = symmetric_eigh(product).eigenvalues()
            for k in range(p):
                largest = max(largest, abs(values.unsafe_get(k)))
        except e:
            raise Error(String("LinAlgError: error_norm spectral eigenvalues failed (", e.kind, ")"))
        squared_norm = largest
    else:
        raise Error("NotImplementedError: Only spectral and frobenius norms are implemented")
    if scaling:
        squared_norm = squared_norm / Float64(p)
    if squared:
        return squared_norm
    return sqrt(squared_norm)


def shrunk_covariance[dtype: DType](emp_cov: NDArray[dtype], shrinkage: Float64 = 0.1) raises -> NDArray[dtype]:
    """Convex combination (1 - a) S + a tr(S)/p I (sklearn `shrunk_covariance`).

    Accepts a (p, p) matrix or a batch (..., p, p); the result has the input
    shape and dtype. Entries must be finite; `shrinkage` must lie in [0, 1].
    """
    check_float_dtype[dtype]()
    if not (shrinkage >= 0.0 and shrinkage <= 1.0):
        raise Error(String("InvalidParameterError: The 'shrinkage' parameter of shrunk_covariance must be a float in the range [0, 1]. Got ", shrinkage, " instead."))
    if emp_cov.ndim < 2:
        raise Error(String("ValueError: shrunk_covariance: expected a 2-D or batched (..., p, p) array, got shape ", shape_text(emp_cov)))
    var p = emp_cov.shape[emp_cov.ndim - 1]
    if emp_cov.shape[emp_cov.ndim - 2] != p or p < 1:
        raise Error(String("ValueError: shrunk_covariance: expected square trailing dimensions, got shape ", shape_text(emp_cov)))
    var shape = List[Int]()
    var batches = 1
    for k in range(emp_cov.ndim):
        shape.append(emp_cov.shape[k])
        if k < emp_cov.ndim - 2:
            batches *= emp_cov.shape[k]
    var result = nm.zeros[dtype](Shape(shape))
    var row_stride = Int(emp_cov.strides[emp_cov.ndim - 2])
    var col_stride = Int(emp_cov.strides[emp_cov.ndim - 1])
    for batch in range(batches):
        # Decode the leading multi-index into a strided base displacement.
        var base = 0
        var remainder = batch
        for k in range(emp_cov.ndim - 3, -1, -1):
            base += (remainder % emp_cov.shape[k]) * Int(emp_cov.strides[k])
            remainder //= emp_cov.shape[k]
        var trace: Float64 = 0
        for i in range(p):
            var value = Float64(emp_cov.unsafe_get(base + i * row_stride + i * col_stride))
            if not isfinite(value):
                raise Error("ValueError: shrunk_covariance: input contains NaN or infinity")
            trace += value
        var mu = trace / Float64(p)
        for i in range(p):
            for j in range(p):
                var value = Float64(emp_cov.unsafe_get(base + i * row_stride + j * col_stride))
                if not isfinite(value):
                    raise Error("ValueError: shrunk_covariance: input contains NaN or infinity")
                var shrunk = (1.0 - shrinkage) * value
                if i == j:
                    shrunk += shrinkage * mu
                result.unsafe_set(batch * p * p + i * p + j, shrunk.cast[dtype]())
    return result^


struct EmpiricalCovarianceEngine[engine_dtype: DType](CovarianceEngine):
    """Concrete family engine: every method delegates to this module's kernels."""

    comptime dtype = Self.engine_dtype

    @staticmethod
    def center(x: NDArray[Self.dtype], assume_centered: Bool) raises -> CenteredData[Self.dtype]:
        return center(x, assume_centered)

    @staticmethod
    def empirical(x: NDArray[Self.dtype], assume_centered: Bool) raises -> NDArray[Self.dtype]:
        return empirical_covariance(x, assume_centered=assume_centered)

    @staticmethod
    def precision(covariance: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        return pinvh(covariance)

    @staticmethod
    def logdet(precision: NDArray[Self.dtype]) raises -> Float64:
        return fast_logdet(precision)

    @staticmethod
    def mahalanobis(x: NDArray[Self.dtype], location: NDArray[Self.dtype], precision: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        return squared_mahalanobis(x, location, precision)

    @staticmethod
    def likelihood(empirical: NDArray[Self.dtype], precision: NDArray[Self.dtype]) raises -> Float64:
        return log_likelihood(empirical, precision)

    @staticmethod
    def error_norm(covariance: NDArray[Self.dtype], comparison: NDArray[Self.dtype], norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return error_norm_value(covariance, comparison, norm, scaling, squared)

    @staticmethod
    def shrink(empirical: NDArray[Self.dtype], shrinkage: Float64 = 0.1) raises -> NDArray[Self.dtype]:
        return shrunk_covariance(empirical, shrinkage)


# ---------------------------------------------------------------------------
# Learned model shared by every covariance estimator (sklearn EmpiricalCovariance methods)
# ---------------------------------------------------------------------------


struct CovarianceModel[dtype: DType](Copyable, Movable):
    """Learned location/covariance/optional precision plus inherited sklearn methods.

    Estimators compose this record instead of inheriting EmpiricalCovariance:
    `install` is `_set_covariance`; `get_precision`, `score`, `mahalanobis`
    and `error_norm` are the inherited public methods. `owner` names the
    estimator in error messages.
    """

    var state: CovarianceState[Self.dtype]
    # Float64 covariance kept when sklearn's reference path is Float64 for
    # Float32 input (np.cov); on-demand precision then uses it, as sklearn does.
    var _covariance64: Optional[NDArray[DType.float64]]
    var _precision64: Optional[NDArray[DType.float64]]

    def __init__(out self):
        check_float_dtype[Self.dtype]()
        self.state = CovarianceState[Self.dtype]()
        self._covariance64 = None
        self._precision64 = None

    def reset(mut self):
        self.state.reset()
        self._covariance64 = None
        self._precision64 = None

    def is_fitted(self) -> Bool:
        return Bool(self.state.covariance_)

    def check_fitted(self, owner: String) raises:
        if not self.is_fitted():
            raise Error(String("NotFittedError: This ", owner, " instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."))

    def check_features(self, x: NDArray[Self.dtype], owner: String) raises:
        """Fitted, 2-D, finite input with the training feature count (validate_data reset=False)."""
        self.check_fitted(owner)
        validate_samples(x, owner)
        var expected = self.state.fitted.n_features_in_.value()
        if self.state.covariance_.value().shape[0] != expected:
            raise Error("ValueError: failed fit left covariance_ with a different feature count; refit before inference")
        if x.shape[1] != expected:
            raise Error(String("ValueError: X has ", x.shape[1], " features, but ", owner, " is expecting ", expected, " features as input."))

    def update_location(mut self, location: NDArray[Self.dtype]) raises:
        """Publish fit's location before numerical work, as sklearn does.

        The previous covariance/precision remain available after a failed refit.
        A first failed fit still rejects covariance-dependent inference.
        """
        self.state.location_ = location.copy()
        self.state.fitted.record_fit(1, location.shape[0])

    def install(mut self, location: NDArray[Self.dtype], covariance: NDArray[Self.dtype], store_precision: Bool, owner: String) raises:
        """Validate and store estimates; compute the pseudoinverse when requested.

        A failure leaves the previous fitted state unchanged.
        """
        try:
            validate_matrix(covariance)
        except e:
            raise Error(String("ValueError: ", owner, ": estimated covariance must be finite: ", e))
        var precision: Optional[NDArray[Self.dtype]] = None
        if store_precision:
            precision = pinvh(covariance)
        self.state.replace(location, covariance, precision^)
        self._covariance64 = None
        self._precision64 = None

    def install_float64(mut self, location: NDArray[Self.dtype], covariance: NDArray[DType.float64], store_precision: Bool,
                        owner: String) raises:
        """Install a covariance whose sklearn reference is Float64 (e.g. np.cov of Float32 data).

        The pseudoinverse uses the Float64 matrix and Float64 cutoff, then both
        are rounded to the estimator dtype. For Float64 this equals `install`.
        """
        try:
            validate_matrix(covariance)
        except e:
            raise Error(String("ValueError: ", owner, ": estimated covariance must be finite: ", e))
        var precision: Optional[NDArray[Self.dtype]] = None
        var precision64: Optional[NDArray[DType.float64]] = None
        if store_precision:
            precision64 = pinvh(covariance)
            precision = to_array[Self.dtype](precision64.value())
        self.state.replace(location, to_array[Self.dtype](covariance), precision^)
        self._covariance64 = covariance.copy()
        self._precision64 = precision64^

    def install(mut self, location: NDArray[Self.dtype], covariance: NDArray[Self.dtype], precision: Optional[NDArray[Self.dtype]]) raises:
        """Store estimates with a precision computed by the estimator (e.g. graphical lasso)."""
        self.state.replace(location, covariance, precision)
        self._covariance64 = None
        self._precision64 = None

    def install_estimate_float64(mut self, location: NDArray[Self.dtype], covariance: NDArray[DType.float64],
                                 precision: NDArray[DType.float64]) raises:
        """Keep solver precision in Float64 for inference, round public snapshots."""
        self.state.replace(location, to_array[Self.dtype](covariance), Optional(to_array[Self.dtype](precision)))
        self._covariance64 = covariance.copy()
        self._precision64 = precision.copy()

    def n_features_in(self, owner: String) raises -> Int:
        if not self.state.fitted.n_features_in_:
            self.check_fitted(owner)
        return self.state.fitted.n_features_in_.value()

    def location(self, owner: String) raises -> NDArray[Self.dtype]:
        if not self.state.location_:
            self.check_fitted(owner)
        return self.state.location_.value().copy()

    def covariance(self, owner: String) raises -> NDArray[Self.dtype]:
        self.check_fitted(owner)
        return self.state.covariance_.value().copy()

    def precision(self, owner: String) raises -> Optional[NDArray[Self.dtype]]:
        """Stored precision_, or None when fitted with store_precision=False."""
        self.check_fitted(owner)
        if self.state.precision_:
            return self.state.precision_.value().copy()
        return None

    def get_precision(self, owner: String) raises -> NDArray[Self.dtype]:
        """Stored precision, or the pseudoinverse of covariance_ recomputed on demand."""
        self.check_fitted(owner)
        if self.state.precision_:
            return self.state.precision_.value().copy()
        if self._covariance64:
            return to_array[Self.dtype](pinvh(self._covariance64.value()))
        return pinvh(self.state.covariance_.value())

    def precision64(self) raises -> NDArray[DType.float64]:
        """Float64 precision for the Float64 reference path (stored or recomputed)."""
        if self._precision64:
            return self._precision64.value().copy()
        return pinvh(self._covariance64.value())

    def score(self, X_test: NDArray[Self.dtype], owner: String) raises -> Float64:
        """Mean Gaussian log-likelihood of X_test under (location_, covariance_)."""
        self.check_features(X_test, owner)
        var n = X_test.shape[0]
        var p = X_test.shape[1]
        var location = self.state.location_.value().copy()
        var shift = nm.zeros[DType.float64](Shape(p))
        for j in range(p):
            shift.unsafe_set(j, Float64(location.unsafe_get(j)))
        # sklearn forms X_test - location_ in the input dtype first.
        var centered = nm.zeros[Self.dtype](Shape(n, p))
        for i in range(n):
            for j in range(p):
                centered.unsafe_set(i * p + j, entry(X_test, i, j).cast[Self.dtype]() - location.unsafe_get(j))
        var test_covariance = to_array[Self.dtype](gram(centered_columns(centered, nm.zeros[DType.float64](Shape(p))), Float64(n)))
        if self._covariance64:
            return log_likelihood(test_covariance, self.precision64())
        return log_likelihood(test_covariance, self.get_precision(owner))

    def mahalanobis(self, X: NDArray[Self.dtype], owner: String) raises -> NDArray[Self.dtype]:
        """Squared Mahalanobis distances (n,) of X to location_ under get_precision()."""
        self.check_features(X, owner)
        if self._covariance64:
            return squared_mahalanobis(X, self.state.location_.value(), self.precision64())
        return squared_mahalanobis(X, self.state.location_.value(), self.get_precision(owner))

    def error_norm(self, comp_cov: NDArray[Self.dtype], owner: String, norm: String = "frobenius", scaling: Bool = True,
                   squared: Bool = True) raises -> Float64:
        self.check_fitted(owner)
        return error_norm_value(self.state.covariance_.value(), comp_cov, norm, scaling, squared)


# ---------------------------------------------------------------------------
# EmpiricalCovariance
# ---------------------------------------------------------------------------


@fieldwise_init
struct EmpiricalCovarianceParams(Copyable, Movable, Writable):
    """Constructor configuration of EmpiricalCovariance (get_params/set_params record)."""

    var store_precision: Bool
    var assume_centered: Bool


struct EmpiricalCovariance[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Maximum likelihood covariance estimator (sklearn `EmpiricalCovariance`).

    Constructor defaults: store_precision=True, assume_centered=False.
    Learned attributes are methods: `location_()`, `covariance_()`,
    `precision_()` (Optional; None when store_precision=False) and
    `n_features_in_()`. `fit` mutates in place and returns nothing.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = EmpiricalCovarianceParams
    comptime NAME = "EmpiricalCovariance"

    var store_precision: Bool
    var assume_centered: Bool
    var _model: CovarianceModel[Self.dtype]

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self._model = CovarianceModel[Self.dtype]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, store_precision: Optional[Bool] = None, assume_centered: Optional[Bool] = None) raises -> ref[self] Self:
        """Update only the supplied parameters."""
        if store_precision:
            self.store_precision = store_precision.value()
        if assume_centered:
            self.assume_centered = assume_centered.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Fit the maximum likelihood covariance; `y` is ignored."""
        validate_samples(X, Self.NAME + ".fit")
        var p = X.shape[1]
        var location: NDArray[Self.dtype]
        if self.assume_centered:
            location = nm.zeros[Self.dtype](Shape(p))
        else:
            location = to_array[Self.dtype](column_means(X))
        self._model.update_location(location)
        if self.assume_centered:
            self._model.install(location, empirical_covariance(X, assume_centered=True), self.store_precision, Self.NAME)
        else:
            # sklearn: np.cov returns Float64 even for Float32 input.
            self._model.install_float64(location, empirical_covariance_float64(X, False), self.store_precision, Self.NAME)
        return self

    def get_precision(self) raises -> Self.Precision:
        return self._model.get_precision(Self.NAME)

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        return self._model.mahalanobis(X, Self.NAME)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        return self._model.score(X, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return self._model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._model.n_features_in(Self.NAME)
