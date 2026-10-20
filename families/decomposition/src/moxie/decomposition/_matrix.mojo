"""Private dense helpers shared by the decomposition estimators.

Element access honours NuMojo strides (including negative ones), and every
reduction accumulates in Float64 so Float32 estimators keep their reference
precision. These are small loops over the accepted NuMojo storage, not a new
array type; factorizations live in `_svd_engine` and mojo-linalg.
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt
from std.utils.numerics import inf
from mojo_linalg import symmetric_indefinite, symmetric_slogdet
from moxie._random import RandomState, entropy_state


@always_inline
def get2[dtype: DType](a: NDArray[dtype], i: Int, j: Int) raises -> Float64:
    """Element (i, j) of a rank-2 array as Float64, honouring strides."""
    return Float64(a.unsafe_get(i * Int(a.strides[0]) + j * Int(a.strides[1])))


@always_inline
def set2[dtype: DType](mut a: NDArray[dtype], i: Int, j: Int, value: Float64) raises:
    """Store value at (i, j) of a rank-2 array, honouring strides."""
    a.unsafe_set(i * Int(a.strides[0]) + j * Int(a.strides[1]), Scalar[dtype](value))


@always_inline
def get1[dtype: DType](a: NDArray[dtype], i: Int) raises -> Float64:
    """Element i of a rank-1 array as Float64, honouring its stride."""
    return Float64(a.unsafe_get(i * Int(a.strides[0])))


@always_inline
def set1[dtype: DType](mut a: NDArray[dtype], i: Int, value: Float64) raises:
    """Store value at element i of a rank-1 array, honouring its stride."""
    a.unsafe_set(i * Int(a.strides[0]), Scalar[dtype](value))


def contiguous[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Owned C-contiguous copy of the logical elements of a rank-2 array."""
    var result = zeros[dtype](Shape(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            set2(result, i, j, get2(x, i, j))
    return result^


def identity[dtype: DType](n: Int) raises -> NDArray[dtype]:
    """Owned (n, n) identity matrix."""
    var result = zeros[dtype](Shape(n, n))
    for i in range(n):
        set2(result, i, i, 1)
    return result^


def column_means[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Column means of a rank-2 array (Float64 accumulation)."""
    var result = zeros[dtype](Shape(x.shape[1]))
    for j in range(x.shape[1]):
        var total: Float64 = 0
        for i in range(x.shape[0]):
            total += get2(x, i, j)
        set1(result, j, total / Float64(x.shape[0]))
    return result^


def column_variances[dtype: DType](x: NDArray[dtype], *, ddof: Int = 0) raises -> NDArray[dtype]:
    """Two-pass column variances with `ddof` like numpy.var(axis=0)."""
    var result = zeros[dtype](Shape(x.shape[1]))
    var n = x.shape[0]
    for j in range(x.shape[1]):
        var total: Float64 = 0
        for i in range(n):
            total += get2(x, i, j)
        var average = total / Float64(n)
        var squares: Float64 = 0
        for i in range(n):
            var d = get2(x, i, j) - average
            squares += d * d
        set1(result, j, squares / Float64(n - ddof))
    return result^


def subtract_row[dtype: DType](x: NDArray[dtype], row: NDArray[dtype]) raises -> NDArray[dtype]:
    """Owned x - row[None, :] for a rank-2 x and a rank-1 row."""
    var result = zeros[dtype](Shape(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            set2(result, i, j, get2(x, i, j) - get1(row, j))
    return result^


def frobenius_squared[dtype: DType](x: NDArray[dtype]) raises -> Float64:
    """Sum of squares of every logical element of a rank-2 array."""
    var total: Float64 = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var v = get2(x, i, j)
            total += v * v
    return total


def seeded_state(random_state: Optional[Int], estimator: String) raises -> RandomState:
    """sklearn check_random_state for native seeds: int resets, None draws entropy."""
    if not random_state:
        return entropy_state()
    var seed = random_state.value()
    if seed < 0 or seed > 4294967295:
        raise Error("ValueError: " + estimator + " random_state must be in [0, 2**32 - 1], got " + String(seed))
    return RandomState(seed)


def machine_epsilon[dtype: DType]() -> Float64:
    """numpy.finfo(dtype).eps for Float32/Float64."""
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-07
    return 2.220446049250313e-16


def feature_names(prefix: String, count: Int, input_features: Optional[List[String]], n_features_in: Int) raises -> List[String]:
    """sklearn ClassNamePrefixFeaturesOutMixin names, validating input names."""
    if input_features and len(input_features.value()) != n_features_in:
        raise Error("ValueError: input_features should have length equal to n_features_in_ ("
            + String(n_features_in) + "), got " + String(len(input_features.value())))
    var names = List[String]()
    for i in range(count):
        names.append(prefix + String(i))
    return names^


def as_float64[dtype: DType](x: NDArray[dtype]) raises -> NDArray[DType.float64]:
    """Owned contiguous Float64 copy of a rank-2 array (numpy astype(float64))."""
    var result = zeros[DType.float64](Shape(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            set2(result, i, j, get2(x, i, j))
    return result^


def symmetric_inverse[dtype: DType](a: NDArray[dtype], what: String) raises -> NDArray[dtype]:
    """Inverse of a symmetric matrix via mojo-linalg's pivoted LDL^T solve.

    Raises a ValueError naming `what` when the matrix is exactly singular.
    """
    var n = a.shape[0]
    try:
        var factor = symmetric_indefinite(a)
        return factor.solve(identity[dtype](n))
    except error:
        raise Error("LinAlgError: cannot invert " + what + ": " + String(error))


def symmetric_logdet[dtype: DType](a: NDArray[dtype]) raises -> Float64:
    """sklearn fast_logdet for symmetric input: log|det| if det > 0, else -inf."""
    var result: Float64
    try:
        var determinant = symmetric_slogdet(a)
        result = determinant.logabsdet if determinant.sign > 0 else -inf[DType.float64]()
    except error:
        raise Error("LinAlgError: log-determinant failed: " + String(error))
    return result
