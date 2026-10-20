from numojo import NDArray
from moxie.decomposition import randomized_svd


def leading_value(x: NDArray[DType.float64]) raises -> Float64:
    var factors = randomized_svd(x, 1, random_state=0)
    return factors.singular_values.unsafe_get(0)
