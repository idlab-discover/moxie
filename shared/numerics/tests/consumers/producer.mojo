import numojo as nm


def data() raises -> nm.NDArray[DType.float64]:
    return nm.ones[DType.float64](nm.Shape(2, 3))
