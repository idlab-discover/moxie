"""Covariance-type dispatch and precision Cholesky factors.

Mirrors sklearn.mixture._gaussian_mixture (pinned revision):
`_compute_precision_cholesky`, `_compute_precision_cholesky_from_precisions`,
`_compute_log_det_cholesky`, `_set_parameters` precisions and the
`_check_precisions*` validators. Full/tied factors are upper triangular U with
precision = U @ U.T, obtained as inv(L).T for covariance = L @ L.T.
Factorizations and triangular solves use mojo-linalg (`cholesky`,
`triangular_solve`, `symmetric_eigh`) on Float64 work matrices; results are
stored in the estimator dtype. Diag/spherical factors are 1/sqrt(covariance).
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, isfinite, abs
from moxie._mixture_linalg import cholesky, triangular_solve, symmetric_eigh, NumericalError
from moxie.mixture._contracts import PrecisionEngine, covariance_shape, check_shape
from moxie.mixture._scalar_math import log64

comptime ILL_DEFINED_COVARIANCE = (
    "ValueError: Fitting the mixture model failed because some components have"
    " ill-defined empirical covariance (for instance caused by singleton or"
    " collapsed samples). Try to decrease the number of components, increase"
    " reg_covar, or scale the input data."
)


def ill_defined_message[dtype: DType]() -> String:
    """The reference error text, with its Float32 accuracy hint."""
    var message = String(ILL_DEFINED_COVARIANCE)
    comptime if dtype == DType.float32:
        message += " The numerical accuracy can also be improved by passing float64 data instead of float32."
    return message


def _load_matrix[dtype: DType](source: NDArray[dtype], offset: Int, d: Int, flip: Bool) raises -> NDArray[DType.float64]:
    """Copy one contiguous d x d block into a Float64 matrix (optionally flipped ud/lr)."""
    var out = zeros[DType.float64](Shape(d, d))
    for i in range(d):
        for j in range(d):
            var value = Float64(source.unsafe_get(offset + i * d + j))
            if flip:
                out.unsafe_set((d - 1 - i) * d + (d - 1 - j), value)
            else:
                out.unsafe_set(i * d + j, value)
    return out^


def _identity(d: Int) raises -> NDArray[DType.float64]:
    var eye = zeros[DType.float64](Shape(d, d))
    for i in range(d):
        eye.unsafe_set(i * d + i, 1.0)
    return eye^


def _upper_precision_factor(cov: NDArray[DType.float64], d: Int) raises NumericalError -> NDArray[DType.float64]:
    """Return inv(L) for cov = L L^T (lower); the caller stores its transpose."""
    var factor = cholesky(cov)
    var lower = factor.lower_factor()
    return triangular_solve(lower, _identity(d), lower=True)


def compute_precision_cholesky[dtype: DType](covariances: NDArray[dtype], kind: String) raises -> NDArray[dtype]:
    """Precision Cholesky factors from covariances (sklearn `_compute_precision_cholesky`).

    Raises the reference ValueError text when a covariance is not positive
    definite (full/tied) or not strictly positive (diag/spherical).
    """
    var out = zeros[dtype](covariances.shape)
    if kind == "full" or kind == "tied":
        var components = covariances.shape[0] if kind == "full" else 1
        var d = covariances.shape[1] if kind == "full" else covariances.shape[0]
        for k in range(components):
            var cov = _load_matrix(covariances, k * d * d, d, False)
            var inverse_lower: NDArray[DType.float64]
            try:
                inverse_lower = _upper_precision_factor(cov, d)
            except error:
                raise Error(ill_defined_message[dtype]())
            for i in range(d):
                for j in range(d):
                    # precisions_chol = solve_triangular(L, I, lower=True).T
                    out.unsafe_set(k * d * d + i * d + j, inverse_lower.unsafe_get(j * d + i).cast[dtype]())
        return out^
    if kind == "diag" or kind == "spherical":
        for i in range(covariances.size):
            var value = Float64(covariances.unsafe_get(i))
            if not (value > 0.0):
                raise Error(ill_defined_message[dtype]())
        for i in range(covariances.size):
            var value = covariances.unsafe_get(i)
            # 1.0 / sqrt(covariances) in the array dtype, as NumPy evaluates it.
            out.unsafe_set(i, 1.0 / sqrt(value))
        return out^
    raise Error(String("ValueError: unsupported covariance_type '", kind, "'"))


def precision_cholesky_from_precisions[dtype: DType](precisions: NDArray[dtype], kind: String) raises -> NDArray[dtype]:
    """Return flipud/lr(cholesky(flipud/lr(P))) per matrix; sqrt for diag/spherical."""
    var out = zeros[dtype](precisions.shape)
    if kind == "full" or kind == "tied":
        var components = precisions.shape[0] if kind == "full" else 1
        var d = precisions.shape[1] if kind == "full" else precisions.shape[0]
        for k in range(components):
            var flipped = _load_matrix(precisions, k * d * d, d, True)
            var lower: NDArray[DType.float64]
            try:
                lower = cholesky(flipped).lower_factor()
            except error:
                raise Error(String("ValueError: '", kind, " precision' should be symmetric, positive-definite"))
            for i in range(d):
                for j in range(d):
                    out.unsafe_set(k * d * d + i * d + j, lower.unsafe_get((d - 1 - i) * d + (d - 1 - j)).cast[dtype]())
        return out^
    for i in range(precisions.size):
        out.unsafe_set(i, sqrt(precisions.unsafe_get(i)))
    return out^


def log_det_cholesky[dtype: DType](factor: NDArray[dtype], kind: String, n_features: Int) raises -> List[Float64]:
    """Log-determinant of each precision Cholesky factor (`_compute_log_det_cholesky`).

    Returns one value per component; tied returns a single shared value.
    Values are rounded through the dtype like the reference arrays.
    """
    var result = List[Float64]()
    if kind == "full":
        var d = factor.shape[1]
        for k in range(factor.shape[0]):
            var total = 0.0
            for i in range(d):
                total += log64(Float64(factor.unsafe_get(k * d * d + i * d + i)))
            result.append(Float64(total.cast[dtype]()))
    elif kind == "tied":
        var d = factor.shape[0]
        var total = 0.0
        for i in range(d):
            total += log64(Float64(factor.unsafe_get(i * d + i)))
        result.append(Float64(total.cast[dtype]()))
    elif kind == "diag":
        var d = factor.shape[1]
        for k in range(factor.shape[0]):
            var total = 0.0
            for j in range(d):
                total += log64(Float64(factor.unsafe_get(k * d + j)))
            result.append(Float64(total.cast[dtype]()))
    elif kind == "spherical":
        for k in range(factor.shape[0]):
            var value = Float64(n_features) * log64(Float64(factor.unsafe_get(k)))
            result.append(Float64(value.cast[dtype]()))
    else:
        raise Error(String("ValueError: unsupported covariance_type '", kind, "'"))
    return result^


def precisions_from_cholesky[dtype: DType](factor: NDArray[dtype], kind: String) raises -> NDArray[dtype]:
    """Return precisions_ = U @ U.T per matrix (full/tied) or factor**2 (diag/spherical)."""
    var out = zeros[dtype](factor.shape)
    if kind == "full" or kind == "tied":
        var components = factor.shape[0] if kind == "full" else 1
        var d = factor.shape[1] if kind == "full" else factor.shape[0]
        for k in range(components):
            var base = k * d * d
            for i in range(d):
                for j in range(d):
                    var total = 0.0
                    for l in range(d):
                        total += Float64(factor.unsafe_get(base + i * d + l)) * Float64(factor.unsafe_get(base + j * d + l))
                    out.unsafe_set(base + i * d + j, total.cast[dtype]())
        return out^
    for i in range(factor.size):
        var value = factor.unsafe_get(i)
        out.unsafe_set(i, value * value)
    return out^


def check_precision_matrix[dtype: DType](matrix: NDArray[DType.float64], label: String) raises:
    """Symmetric (np.isclose defaults) and positive definite (eigvalsh > 0)."""
    var d = matrix.shape[0]
    for i in range(d):
        for j in range(d):
            var a = matrix.unsafe_get(i * d + j)
            var b = matrix.unsafe_get(j * d + i)
            if not isfinite(a) or abs(a - b) > 1e-8 + 1e-5 * abs(b):
                raise Error(String("ValueError: '", label, " precision' should be symmetric, positive-definite"))
    var positive = True
    try:
        var eigen = symmetric_eigh(matrix)
        var values = eigen.eigenvalues()
        for i in range(d):
            if not (values.unsafe_get(i) > 0.0):
                positive = False
    except error:
        positive = False
    if not positive:
        raise Error(String("ValueError: '", label, " precision' should be symmetric, positive-definite"))


def check_precisions[dtype: DType](precisions: NDArray[dtype], kind: String, components: Int, features: Int) raises:
    """Validate precisions_init shape and definiteness (sklearn `_check_precisions`)."""
    var expected = covariance_shape(kind, components, features)
    if precisions.ndim != len(expected):
        raise Error(String("ValueError: The parameter '", kind, " precision' should have the shape of ", expected, ", but got ", precisions.shape))
    for axis in range(precisions.ndim):
        if precisions.shape[axis] != expected[axis]:
            raise Error(String("ValueError: The parameter '", kind, " precision' should have the shape of ", expected, ", but got ", precisions.shape))
    var compact = _compact(precisions)
    for i in range(compact.size):
        if not isfinite(Float64(compact.unsafe_get(i))):
            raise Error("ValueError: Input contains NaN or infinity in precisions_init")
    if kind == "full" or kind == "tied":
        var count = components if kind == "full" else 1
        for k in range(count):
            check_precision_matrix[dtype](_load_matrix(compact, k * features * features, features, False), kind)
    else:
        for i in range(compact.size):
            if not (Float64(compact.unsafe_get(i)) > 0.0):
                raise Error(String("ValueError: '", kind, " precision' should be positive"))


def _compact[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Row-major logical copy of a 1-3 dimensional array (any strides)."""
    var out = zeros[dtype](x.shape)
    if x.ndim == 1:
        for i in range(x.shape[0]):
            out.unsafe_set(i, x.unsafe_get(i * Int(x.strides[0])))
    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out.unsafe_set(i * x.shape[1] + j, x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
    elif x.ndim == 3:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for l in range(x.shape[2]):
                    out.unsafe_set((i * x.shape[1] + j) * x.shape[2] + l,
                                   x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]) + l * Int(x.strides[2])))
    else:
        raise Error("ValueError: expected an array of rank 1 to 3")
    return out^


def compact[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Public alias of the row-major logical copy used for owned snapshots."""
    return _compact(x)


@fieldwise_init
struct CholeskyPrecision[dtype_: DType](PrecisionEngine, Copyable):
    """PrecisionEngine over the functions above (family extension point)."""

    comptime dtype = Self.dtype_

    def from_covariances(self, covariance: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        return compute_precision_cholesky(_compact(covariance), kind)

    def from_precisions(self, precision: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        return precision_cholesky_from_precisions(_compact(precision), kind)
