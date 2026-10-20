"""Consumer fixture; uses the same pinned family owner as producer."""
from numojo import NDArray
from moxie._dense import sum_axis2d
from moxie.semi_supervised._contracts import validate_probability_shape

def consume(values: NDArray[DType.float64]) raises -> Float64:
    validate_probability_shape(values, 2, 2)
    var totals = sum_axis2d(values, 1)
    return Float64(totals.unsafe_get(0))
