"""Nonlinear embedding and a learned pre-image map."""
from numojo import Shape, zeros
from moxie.decomposition import KernelPCA


def main() raises:
    var x = zeros[DType.float64](Shape(8, 2))
    for i in range(8):
        x.unsafe_set(i * 2, Float64(i) / 7)
        x.unsafe_set(i * 2 + 1, Float64(i * i) / 49)
    var model = KernelPCA(2, kernel="rbf", gamma=2.0, fit_inverse_transform=True)
    var z = model.fit_transform(x)
    var recovered = model.inverse_transform(z)
    print("embedding:", z.shape[0], z.shape[1])
    print("reconstruction:", recovered.shape[0], recovered.shape[1])
