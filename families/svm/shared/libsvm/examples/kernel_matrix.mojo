"""Evaluate a dense kernel; fitting is not implemented in this project yet."""
import numojo as nm
from moxie.svm._contracts import KernelSpec
from moxie.svm._kernels import kernel_matrix, resolve_gamma


def main() raises:
    var x = nm.arange[DType.float64](6).reshape(nm.Shape(3, 2))
    var gamma = resolve_gamma(x, "scale")
    var gram = kernel_matrix(x, x, KernelSpec("rbf", gamma, 3, 0))
    print("RBF Gram shape:", gram.shape[0], gram.shape[1])
    print("First diagonal:", gram.item(0, 0))
