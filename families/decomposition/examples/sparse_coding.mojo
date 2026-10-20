"""Dense coefficients after soft thresholding against fixed dictionary atoms."""
from numojo import Shape, zeros
from moxie.decomposition import SparseCoder


def main() raises:
    var d = zeros[DType.float64](Shape(2, 3))
    d.unsafe_set(0, 1)
    d.unsafe_set(4, 1)
    var x = zeros[DType.float64](Shape(1, 3))
    x.unsafe_set(0, 3)
    x.unsafe_set(1, -2)
    var coder = SparseCoder(d, transform_algorithm="threshold", transform_alpha=0.5)
    var code = coder.transform(x)
    print("code:", code.unsafe_get(0), code.unsafe_get(1))
