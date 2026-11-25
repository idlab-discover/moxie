"""KernelRidge adapter for GP kernels with a two-dimensional covariance."""
import numojo as nm
from moxie.gaussian_process.kernels import Kernel, KernelLike
from ._api import KernelParameters
from ._kernels import MatrixKernel
from ._promotion import FloatingArray


struct GaussianProcessKernel(Copyable, MatrixKernel):
    """Use a typed GP leaf or composed GP kernel in KernelRidge."""

    var kernel: Kernel

    def __init__(out self, var kernel: Kernel):
        self.kernel = kernel^

    def __init__[K: KernelLike](out self, kernel: K):
        self.kernel = kernel.as_kernel()

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
        return FloatingArray(self.kernel(X, Y^))
