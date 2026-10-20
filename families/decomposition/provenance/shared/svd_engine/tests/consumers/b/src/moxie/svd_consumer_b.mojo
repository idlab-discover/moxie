from numojo import NDArray
from moxie.decomposition._svd_engine import dense_svd


def leading_value(x: NDArray[DType.float64]) raises -> Float64:
    var factors = dense_svd(x)
    return factors.singular_values.unsafe_get(0)
