"""Numerical backend contract for dense PCA, independent of estimator state."""
from numojo import NDArray, Shape, zeros
from mojo_linalg import svd, symmetric_eigh, symmetric_indefinite


trait PCABackend:
    """Float32/Float64 kernels over owned NuMojo outputs.

    full returns thin (U, s, Vt), with s descending. eigh returns ascending
    eigenvalues and column eigenvectors. Inputs are borrowed and preserved.
    validate_fit enforces backend-specific training dimensions before fitting.
    """

    @staticmethod
    def validate_fit[dtype: DType](X: NDArray[dtype]) raises: ...

    @staticmethod
    def full[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], NDArray[dtype]]: ...

    @staticmethod
    def eigh[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]: ...

    @staticmethod
    def inverse[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]: ...

    @staticmethod
    def slogdet[dtype: DType](X: NDArray[dtype]) raises -> Tuple[Float64, Float64]: ...

    @staticmethod
    def matmul[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]: ...


struct RuntimePCABackend(PCABackend):
    @staticmethod
    def validate_fit[dtype: DType](X: NDArray[dtype]) raises:
        pass

    @staticmethod
    def full[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype], NDArray[dtype]]:
        var factors = svd(X)
        return (factors.u(), factors.singular_values(), factors.vt())

    @staticmethod
    def eigh[dtype: DType](X: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
        var factors = symmetric_eigh(X)
        return (factors.eigenvalues(), factors.eigenvectors())

    @staticmethod
    def inverse[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
        var identity = zeros[dtype](Shape(X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            identity.unsafe_set(i * X.shape[0] + i, 1)
        var factors = symmetric_indefinite(X)
        return factors.solve(identity)

    @staticmethod
    def slogdet[dtype: DType](X: NDArray[dtype]) raises -> Tuple[Float64, Float64]:
        var factors = symmetric_indefinite(X)
        var determinant = factors.slogdet()
        return (Float64(determinant.sign), determinant.logabsdet)

    @staticmethod
    def matmul[dtype: DType](X: NDArray[dtype], Y: NDArray[dtype]) raises -> NDArray[dtype]:
        return X @ Y
