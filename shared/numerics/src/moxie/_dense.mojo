"""Small native operations on NuMojo storage, with explicit 2-D contracts."""
import numojo as nm


def sum_axis2d[
    dtype: DType
](x: nm.NDArray[dtype], axis: Int) raises -> nm.NDArray[dtype]:
    """Sum a matrix axis, preserving dtype; widen accumulation to Float64."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2 or axis < -2 or axis > 1:
        raise Error("ValueError: expected a matrix and axis in [-2, 1]")
    var dim = axis + 2 if axis < 0 else axis
    var count = x.shape[1 - dim]
    var reduced = x.shape[dim]
    var outer_stride = Int(x.strides[1 - dim])
    var inner_stride = Int(x.strides[dim])
    # Bounds follow from validated rank and the loop extents. NuMojo owns
    # offset/lifetime; unsafe_get receives an element displacement, not bytes.
    var output = nm.zeros[dtype](nm.Shape(count))
    for i in range(count):
        var total = Float64(0)
        for j in range(reduced):
            total += Float64(x.unsafe_get(i * outer_stride + j * inner_stride))
        output.unsafe_set(i, total.cast[dtype]())
    return output^


def row_squared_norms[
    dtype: DType
](x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    """Dense squared norms with no matrix-sized intermediate; not sparse dispatch.
    """
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected a matrix")
    # Bounds follow from validated rank and the loop extents. NuMojo owns
    # offset/lifetime; unsafe_get receives an element displacement, not bytes.
    var rows = x.shape[0]
    var cols = x.shape[1]
    var row_stride = Int(x.strides[0])
    var col_stride = Int(x.strides[1])
    var output = nm.zeros[dtype](nm.Shape(rows))
    for i in range(rows):
        var total = Float64(0)
        for j in range(cols):
            var value = Float64(x.unsafe_get(i * row_stride + j * col_stride))
            total += value * value
        output.unsafe_set(i, total.cast[dtype]())
    return output^
