"""Run with pixi run numax-start."""
from numojo import Shape, zeros
from moxie.decomposition import PCA
from moxie_numax import NumaxPCABackend


def main() raises:
    var X = zeros[DType.float64](Shape(6, 2))
    for i in range(6):
        X.unsafe_set(2 * i, Float64(i))
        X.unsafe_set(2 * i + 1, Float64(i * i))
    var model = PCA[DType.float64, NumaxPCABackend[6, 2]](1, svd_solver="full")
    var projected = model.fit_transform(X)
    print("Singular values:", model.singular_values_.value())
    print("Projection:", projected)
    print("Reconstruction:", model.inverse_transform(projected))
