"""An owned rank-two approximation of a dense observation matrix."""
from numojo import zeros, Shape
from moxie.decomposition import randomized_svd


def main() raises:
    var x = zeros[DType.float64](Shape(5, 3))
    for i in range(15):
        x.unsafe_set(i, Float64(i % 7 - 3))
    var result = randomized_svd(x, 2, random_state=42)
    print("Two leading singular values:", result.singular_values)
    print("U shape:", result.U.shape, "Vt shape:", result.Vt.shape)
