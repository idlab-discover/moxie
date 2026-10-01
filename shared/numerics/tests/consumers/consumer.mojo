import numojo as nm
from moxie._dense import row_squared_norms


def consume(mut data: nm.NDArray[DType.float64]) raises -> Float64:
    data[nm.Item(0, 0)] = 2
    return row_squared_norms(data).item(0)
