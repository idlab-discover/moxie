import numojo as nm
from std.sys import argv
from moxie.svm._contracts import KernelSpec
from moxie.svm._kernels import kernel_matrix, resolve_gamma


def main() raises:
    var args = argv()
    var rows = Int(args[1])
    var cols = Int(args[2])
    var kind = args[3]
    var owner = nm.zeros[DType.float64](nm.Shape(rows, 2 * cols))
    for i in range(rows):
        for j in range(2 * cols):
            owner.unsafe_set(i * 2 * cols + j, Float64((i * 17 + j * 3) % 31 - 15) / 97)
    var x = owner.view_with_layout(nm.Shape(rows, cols), nm.NDArrayStrides(2 * cols, -2), 2 * cols - 1)
    var result = kernel_matrix(x, x, KernelSpec(kind, gamma=0.7, degree=3, coef0=0.2))
    for i in range(result.size):
        print("kernel", result.unsafe_get(i))
    print("gamma", resolve_gamma(x, "scale"))
