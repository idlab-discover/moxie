"""Native candidate outputs for independent differential comparison."""
import numojo as nm
from std.sys import argv
from moxie._dense import sum_axis2d, row_squared_norms


def emit[dtype: DType](name: String, data: nm.NDArray[dtype]) raises:
    print("shape", name, data.ndim, data.size)
    var flat = data.flatten()
    for i in range(flat.size):
        print(name, i, flat.item(i))


def evaluate[dtype: DType](rows: Int, cols: Int, layout: String) raises:
    var owner = nm.zeros[dtype](nm.Shape(rows, cols * 2))
    for i in range(rows):
        for j in range(cols * 2):
            owner[nm.Item(i, j)] = (
                SIMD[dtype, 1]((i * 17 + j * 3) % 31 - 15) / 97
            )
    var x = owner.view_with_layout(
        nm.Shape(rows, cols), nm.NDArrayStrides(cols * 2, 2), 0
    )
    if layout == "O":
        x = owner.view_with_layout(
            nm.Shape(rows, cols), nm.NDArrayStrides(cols * 2, 2), 1
        )
    elif layout == "R":
        x = owner.view_with_layout(
            nm.Shape(rows, cols), nm.NDArrayStrides(cols * 2, -2), cols * 2 - 2
        )
    elif layout == "C":
        x = x.contiguous()
    elif layout == "T":
        x = x.T()
    emit("input", x)
    print("total", 0, nm.sum(x))
    emit("sum0", sum_axis2d(x, axis=0))
    emit("mean0", nm.mean[returned_dtype=dtype](x, axis=0))
    emit("center", x - nm.mean[returned_dtype=dtype](x, axis=0))
    emit("gram", x @ x.T())
    var weights = nm.zeros[dtype](nm.Shape(x.shape[1], 5))
    for i in range(x.shape[1]):
        for j in range(5):
            weights[nm.Item(i, j)] = (
                SIMD[dtype, 1]((i * 7 + j * 11) % 23 - 11) / 53
            )
    emit("linear", x @ weights)
    # Axis-1 reduction is deliberately evaluated rather than presumed supported.
    emit("norm", row_squared_norms(x))


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("usage: evaluate DTYPE ROWS COLS LAYOUT")
    if args[1] == "f32":
        evaluate[DType.float32](Int(args[2]), Int(args[3]), args[4])
    else:
        evaluate[DType.float64](Int(args[2]), Int(args[3]), args[4])
