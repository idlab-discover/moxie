"""Dense preprocessing validation; no estimator computation or storage wrapper."""
from std.math import isfinite, isnan
from numojo import NDArray


def validate_dense[dtype: DType](
    x: NDArray[dtype], *, allow_nan: Bool = False,
    min_samples: Int = 1, min_features: Int = 1,
) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if min_samples < 0 or min_features < 0:
        raise Error("ValueError: minimum dimensions must be nonnegative")
    if x.ndim != 2:
        raise Error("ValueError: expected a dense matrix")
    if x.shape[0] < min_samples or x.shape[1] < min_features:
        raise Error("ValueError: insufficient samples or features")
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            var value = x.unsafe_get(row * Int(x.strides[0]) + col * Int(x.strides[1]))
            if not isfinite(value):
                if not (allow_nan and isnan(value)):
                    raise Error("ValueError: input contains disallowed NaN or infinity")


def missing_at[dtype: DType](x: NDArray[dtype], row: Int, col: Int) raises -> Bool:
    """Lazy numeric missing mask: True means NaN. No allocation or retained alias."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: missing mask requires a matrix")
    if row < 0 or col < 0 or row >= x.shape[0] or col >= x.shape[1]:
        raise Error("ValueError: missing-mask index out of bounds")
    return isnan(x.unsafe_get(row * Int(x.strides[0]) + col * Int(x.strides[1])))


def require_dense_output(sparse_output: Bool) raises:
    if sparse_output:
        raise Error("ValueError: sparse output is deferred; select sparse_output=False")


def validate_category_code(code: Int, size: Int, *, allow_unknown: Bool = False) raises:
    """Check one vocabulary index; -1 is a private unknown sentinel, never a class."""
    if size < 0:
        raise Error("ValueError: negative vocabulary size")
    if allow_unknown and code == -1:
        return
    if code < 0 or code >= size:
        raise Error("ValueError: category code outside vocabulary")
