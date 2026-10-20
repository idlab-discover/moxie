"""Blind source separation of two mixed signals with FastICA."""
from std.math import sin
from numojo import Shape, zeros
from moxie.decomposition import FastICA


def main() raises:
    var n = 200
    var x = zeros[DType.float64](Shape(n, 2))
    for i in range(n):
        var t = Float64(i) / 20
        var s1 = sin(2 * t)                         # sine source
        var s2 = 1.0 if Int(t) % 2 == 0 else -1.0   # square-wave source
        x.unsafe_set(i * 2, s1 + 0.5 * s2)
        x.unsafe_set(i * 2 + 1, 0.4 * s1 + s2)
    var ica = FastICA(n_components=2, random_state=0)
    var sources = ica.fit_transform(x)
    print("sources shape:", sources.shape[0], sources.shape[1])
    print("iterations:", ica.n_iter_)
    print("estimated mixing matrix:", ica.mixing_.value())
