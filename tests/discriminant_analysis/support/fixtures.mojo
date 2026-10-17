"""Test-only helpers: array construction/comparison and native covariance estimators."""
from std.math import isnan, abs
from numojo import NDArray, Shape, zeros
from moxie.discriminant_analysis import CovarianceEstimator
from moxie._discriminant_analysis._covariance import (
    empirical_covariance,
    shrunk_covariance,
    ledoit_wolf,
    standardized_ledoit_wolf,
)


struct FixtureCovariance(CovarianceEstimator):
    """User-style covariance estimator selected by name (mirrors the parity oracle).

    kinds: 'shrunk' (ShrunkCovariance), 'lw' (LedoitWolf), 'stdlw' (standardized
    Ledoit-Wolf test helper), 'diag' (diagonal variances), 'empirical'.
    """

    var kind: String
    var shrinkage: Float64
    var fitted: Optional[NDArray[DType.float64]]
    var fit_calls: Int

    def __init__(out self, kind: String, shrinkage: Float64 = 0.1):
        self.kind = kind
        self.shrinkage = shrinkage
        self.fitted = None
        self.fit_calls = 0

    def fit(mut self, X: NDArray[DType.float64]) raises:
        self.fit_calls += 1
        if self.kind == "shrunk":
            self.fitted = shrunk_covariance(empirical_covariance(X), self.shrinkage)
        elif self.kind == "lw":
            self.fitted = ledoit_wolf(X)
        elif self.kind == "stdlw":
            self.fitted = standardized_ledoit_wolf(X)
        elif self.kind == "empirical":
            self.fitted = empirical_covariance(X)
        elif self.kind == "diag":
            var c = empirical_covariance(X)
            var p = c.shape[0]
            for i in range(p):
                for j in range(p):
                    if i != j:
                        c.unsafe_set(i * p + j, 0)
            self.fitted = c^
        else:
            raise Error("ValueError: unknown fixture covariance kind " + self.kind)

    def covariance(self) raises -> NDArray[DType.float64]:
        if not self.fitted:
            raise Error("NotFittedError: fixture covariance estimator is not fitted")
        return self.fitted.value().copy()


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    """Row-major (rows, cols) array from values."""
    var a = zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, SIMD[dtype, 1](values[i]))
    return a^


def parse_numbers(text: String) raises -> List[Float64]:
    var out = List[Float64]()
    if text.byte_length() == 0:
        return out^
    for token in text.split(","):
        out.append(Float64(String(token)))
    return out^


def parse_array[dtype: DType](text: String) raises -> NDArray[dtype]:
    """Decode 'rows:cols:v,v,...' (cols == -1 means a 1-D array)."""
    var fields = text.split(":")
    var rows = Int(String(fields[0]))
    var cols = Int(String(fields[1]))
    var values = parse_numbers(String(fields[2]))
    var a: NDArray[dtype]
    if cols < 0:
        a = zeros[dtype](Shape(rows))
    else:
        a = zeros[dtype](Shape(rows, cols))
    if len(values) != a.size:
        raise Error("fixture array size mismatch: " + text)
    for i in range(len(values)):
        a.unsafe_set(i, SIMD[dtype, 1](values[i]))
    return a^


def max_abs[dtype: DType](a: NDArray[dtype]) -> Float64:
    var m: Float64 = 0
    for i in range(a.size):
        var v = abs(Float64(a.unsafe_get(i)))
        if v > m:
            m = v
    return m


def assert_close[
    dtype: DType
](actual: NDArray[dtype], expected: NDArray[DType.float64], tol: Float64, label: String) raises:
    """Norm-wise check |a - b| <= tol * max(1, max|b|); NaNs must match."""
    if actual.ndim != expected.ndim:
        raise Error(label + ": rank " + String(actual.ndim) + " != " + String(expected.ndim))
    for d in range(actual.ndim):
        if actual.shape[d] != expected.shape[d]:
            raise Error(label + ": shape " + String(actual.shape) + " != " + String(expected.shape))
    var bound = tol * max(1.0, max_abs(expected))
    for i in range(actual.size):
        var a = Float64(actual.unsafe_get(i))
        var b = expected.unsafe_get(i)
        if isnan(a) and isnan(b):
            continue
        if a == b:
            continue
        if not (abs(a - b) <= bound):
            raise Error(
                label + ": element " + String(i) + " actual=" + String(a) + " expected=" + String(b)
                + " bound=" + String(bound)
            )


def assert_columns_close_up_to_sign[
    dtype: DType
](
    actual: NDArray[dtype], expected: NDArray[DType.float64], columns: Int, tol: Float64, label: String
) raises:
    """Compare the first `columns` columns, each up to a sign flip."""
    if actual.ndim != 2 or expected.ndim != 2 or actual.shape[0] != expected.shape[0] or actual.shape[1] != expected.shape[1]:
        raise Error(label + ": shape " + String(actual.shape) + " != " + String(expected.shape))
    var rows = actual.shape[0]
    var cols = actual.shape[1]
    var bound = tol * max(1.0, max_abs(expected))
    for j in range(min(columns, cols)):
        var dot: Float64 = 0
        for i in range(rows):
            dot += Float64(actual.unsafe_get(i * cols + j)) * expected.unsafe_get(i * cols + j)
        var sign: Float64 = -1.0 if dot < 0 else 1.0
        for i in range(rows):
            var a = sign * Float64(actual.unsafe_get(i * cols + j))
            var b = expected.unsafe_get(i * cols + j)
            if not (abs(a - b) <= bound):
                raise Error(
                    label + ": column " + String(j) + " row " + String(i) + " actual=" + String(a)
                    + " expected=" + String(b) + " bound=" + String(bound)
                )
