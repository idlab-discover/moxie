from numojo import zeros, Shape
from moxie.manifold import ClassicalMDS, trustworthiness


def main() raises:
    var x = zeros[DType.float64](Shape(7, 3))
    for i in range(7):
        x.unsafe_set(i * 3, Float64(i))
        x.unsafe_set(i * 3 + 1, Float64(i * i))
        x.unsafe_set(i * 3 + 2, Float64(i % 3))
    var model = ClassicalMDS(2)
    var coordinates = model.fit_transform(x)
    print("Embedded samples:", coordinates.shape[0])
    print("Components:", coordinates.shape[1])
    print("Trustworthiness:", trustworthiness(x, coordinates, n_neighbors=2))
