"""Nystroem feature maps with a named RBF kernel and a typed callable kernel."""
from numojo import NDArray, Shape
from moxie.kernel_approximation import Nystroem
from moxie._kernel_ridge import RowKernel


@fieldwise_init
struct ScaledDot(Copyable, Movable, RowKernel):
    var scale: Float64

    def evaluate[dtype: DType](self, X: NDArray[dtype], i: Int, Y: NDArray[dtype], j: Int) raises -> SIMD[dtype, 1]:
        var total = Float64(0)
        for k in range(X.shape[1]):
            total += Float64(X.unsafe_get(i * X.shape[1] + k)) * Float64(Y.unsafe_get(j * Y.shape[1] + k))
        return SIMD[dtype, 1](self.scale * total)


def main() raises:
    var x = NDArray[DType.float64](Shape(5, 2))
    for k in range(10):
        x.unsafe_set(k, Float64(k % 3) / 2.0)
    var rbf = Nystroem(gamma=0.2, n_components=3, random_state=1)
    var features = rbf.fit_transform(x)
    print("RBF Nystroem output:", features.shape[0], features.shape[1], "basis rows:", rbf.component_indices_.size)
    var custom = Nystroem[DType.float64, ScaledDot](ScaledDot(0.5), n_components=3, random_state=1)
    print("Callable Nystroem output:", custom.fit_transform(x).shape[1])
