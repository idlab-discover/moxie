"""KernelRidge adapters for GP kernels with a two-dimensional covariance."""
import numojo as nm

from moxie.gaussian_process._kernel_math import as_dense64
from moxie.gaussian_process._plumbing import Dense64, KernelContract
from moxie.gaussian_process.kernels import Kernel, KernelLike

from ._api import KernelParameters
from ._kernels import MatrixKernel
from ._promotion import FloatingArray


comptime GaussianProcessKernel = TypedGaussianProcessKernel[Kernel]


struct TypedGaussianProcessKernel[K: KernelContract](Copyable, MatrixKernel):
    """Use a dense GP kernel as a matrix kernel in KernelRidge."""

    var kernel: Self.K

    def __init__(out self, var kernel: Self.K):
        comptime assert (
            Self.K.Input == Dense64
        ), "KernelRidge GP kernels require Float64 NDArray input"
        self.kernel = kernel^

    def __init__[Leaf: KernelLike](out self, kernel: Leaf):
        comptime assert Self.K == Kernel, "built-in GP leaves use GaussianProcessKernel"
        self.kernel = rebind_var[Self.K](kernel.as_kernel())

    def evaluate_matrix[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        var Y: nm.NDArray[dtype],
        same_data: Bool,
        params: Optional[KernelParameters],
    ) raises -> FloatingArray:
        if params and len(params.value()) != 0:
            raise Error("TypeError: set parameters on the GP kernel object")
        # sklearn's pairwise dispatcher supplies explicit Y even during fit.
        comptime if Self.K == Kernel:
            return FloatingArray(rebind[Kernel](self.kernel)(X, Y^))
        else:
            var left = rebind_var[Self.K.Input](as_dense64(X, "KernelRidge kernel X"))
            var right = rebind_var[Self.K.Input](as_dense64(Y, "KernelRidge kernel Y"))
            var result = self.kernel.evaluate(left, right^)
            return FloatingArray(result^.take_covariance())
