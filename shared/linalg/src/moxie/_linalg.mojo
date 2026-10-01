"""Validated covariance solves over shared NuMojo storage."""
from std.math import isfinite
from numojo import NDArray, eye
from mojo_linalg import CholeskyFactor, cholesky


def _validate_covariance[dtype: DType](covariance: NDArray[dtype]) raises -> Int:
    comptime assert dtype == DType.float32 or dtype == DType.float64, "covariance solves require Float32 or Float64"
    if covariance.ndim != 2:
        raise Error("ValueError: covariance must be a rank-two square matrix")
    var n = covariance.shape[0]
    if covariance.shape[1] != n:
        raise Error("ValueError: covariance must be a rank-two square matrix")
    var rows = Int(covariance.strides[0])
    var cols = Int(covariance.strides[1])
    # The kernel reads one triangle; validate both so covariance callers cannot
    # accidentally hide asymmetry or nonfinite values in the other triangle.
    for i in range(n):
        for j in range(i + 1):
            var lower = covariance.unsafe_get(i * rows + j * cols)
            var upper = covariance.unsafe_get(j * rows + i * cols)
            if not isfinite(lower) or not isfinite(upper):
                raise Error("ValueError: covariance must contain only finite values")
            if lower != upper:
                raise Error("ValueError: covariance must be symmetric")
    return n


struct CovarianceFactor[dtype: DType](Copyable, Movable):
    """Owned SPD factor, with independent outputs and no retained input aliases."""
    var _factor: CholeskyFactor[Self.dtype]
    var _order: Int

    def __init__(out self, covariance: NDArray[Self.dtype]) raises:
        self._order = _validate_covariance(covariance)
        try:
            self._factor = cholesky(covariance)
        except error:
            raise Error("ValueError: covariance factorization failed: " + String(error))

    def order(self) -> Int:
        return self._order

    def lower_factor(self) -> NDArray[Self.dtype]:
        """Independent L snapshot, covariance = L @ L.T."""
        return self._factor.lower_factor()

    def solve(self, rhs: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Solve covariance @ X = rhs; preserve vector/matrix shape and dtype."""
        try:
            return self._factor.solve(rhs)
        except error:
            raise Error("ValueError: covariance solve failed: " + String(error))

    def inverse(self) raises -> NDArray[Self.dtype]:
        """Solve against an identity matrix; no generic inversion kernel."""
        var result = eye[Self.dtype](self._order, self._order)
        try:
            self._factor.solve_inplace(result)
        except error:
            raise Error("ValueError: covariance inverse failed: " + String(error))
        return result^


def covariance_factor[dtype: DType](covariance: NDArray[dtype]) raises -> CovarianceFactor[dtype]:
    return CovarianceFactor[dtype](covariance)


def covariance_solve[dtype: DType](covariance: NDArray[dtype], rhs: NDArray[dtype]) raises -> NDArray[dtype]:
    var factor = covariance_factor(covariance)
    return factor.solve(rhs)


def covariance_inverse[dtype: DType](covariance: NDArray[dtype]) raises -> NDArray[dtype]:
    var factor = covariance_factor(covariance)
    return factor.inverse()
