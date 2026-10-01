"""Validation-only interchange for runtime-sized covariance cases."""
from std.sys import argv
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._linalg import covariance_factor


def matrix[dtype: DType](rows: Int, cols: Int, layout: Int) raises -> NDArray[dtype]:
    if layout == 0 or rows == 0 or cols == 0:
        return nm.zeros[dtype](Shape(rows, cols))
    if layout == 1:
        var backing = nm.zeros[dtype](Shape(cols, rows))
        return backing.view_with_layout(Shape(rows, cols), nm.NDArrayStrides(1, rows), 0)
    var backing = nm.zeros[dtype](Shape(rows * cols * 2 + 8))
    if layout == 2:
        return backing.view_with_layout(Shape(rows, cols), nm.NDArrayStrides(-2 * cols, 2), 3 + (rows - 1) * cols * 2)
    return backing.view_with_layout(Shape(rows, cols), nm.NDArrayStrides(2 * cols, 2), 3)


def run[dtype: DType](data: String) raises:
    var fields = data.strip().split(" ")
    var n = Int(String(fields[1]))
    var cols = Int(String(fields[2]))
    var layout = Int(String(fields[3]))
    var rhs_cols = 1 if cols == -1 else cols
    if n < 0 or n > 512 or cols < -1 or cols > 512 or len(fields) != 4 + n * n + n * rhs_cols:
        raise Error("invalid validation fixture")
    var a = matrix[dtype](n, n, layout)
    var b: NDArray[dtype]
    if cols == -1:
        var backing = nm.zeros[dtype](Shape(n * 2 + 8))
        b = backing.view_with_layout(Shape(n), nm.NDArrayStrides(-2), 3 + max(0, n - 1) * 2)
    else:
        b = matrix[dtype](n, cols, layout)
    for i in range(n):
        for j in range(n):
            a[Item(i, j)] = Scalar[dtype](Float64(String(fields[4 + i * n + j])))
        for j in range(rhs_cols):
            var value = Scalar[dtype](Float64(String(fields[4 + n * n + i * rhs_cols + j])))
            if cols == -1:
                b[Item(i)] = value
            else:
                b[Item(i, j)] = value
    try:
        var factor = covariance_factor(a)
        var x = factor.solve(b)
        var precision = factor.inverse()
        var lower = factor.lower_factor()
        print("OK")
        print(x.ndim)
        for i in range(x.size):
            print(Float64(x.unsafe_get(i)))
        for i in range(precision.size):
            print(Float64(precision.unsafe_get(i)))
        for i in range(lower.size):
            print(Float64(lower.unsafe_get(i)))
    except error:
        print("ERROR")
        print(String(error))


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as stream:
        data = stream.read()
    if data.startswith("32 "):
        run[DType.float32](data)
    else:
        run[DType.float64](data)
