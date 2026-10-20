"""Dense PCA, with runtime-sized NuMojo data and an unfitted clone."""
from numojo import Shape, zeros
from moxie.base import clone
from moxie.decomposition import PCA


def main() raises:
    var X = zeros[DType.float64](Shape(4, 2))
    for i in range(4):
        X.unsafe_set(2 * i, Float64(i))
        X.unsafe_set(2 * i + 1, Float64(i * i))
    var model = PCA(n_components=1)
    var projected = model.fit_transform(X)
    print("components:", model.n_components_)
    print("explained variance:", model.explained_variance_.value().unsafe_get(0))
    print("first projected row:", projected.unsafe_get(0))
    var unfitted = clone(model)
    print("clone fitted:", unfitted.__sklearn_is_fitted__())
