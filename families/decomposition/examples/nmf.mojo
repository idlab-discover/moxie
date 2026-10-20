"""Parts-based factorization of a small non-negative matrix with NMF."""
from numojo import Shape, zeros
from moxie.decomposition import NMF, MiniBatchNMF


def main() raises:
    var x = zeros[DType.float64](Shape(10, 6))
    for i in range(10):
        for j in range(6):
            var part_a = Float64(i % 3) * (1.0 if j < 3 else 0.0)
            var part_b = Float64(i % 4) * (1.0 if j >= 3 else 0.2)
            x.unsafe_set(i * 6 + j, part_a + part_b)
    var nmf = NMF(n_components=2, init=Optional[String]("nndsvda"), random_state=0)
    var w = nmf.fit_transform(x)
    print("W shape:", w.shape[0], w.shape[1], "iterations:", nmf.n_iter_)
    print("reconstruction error:", nmf.reconstruction_err_)
    var online = MiniBatchNMF(n_components=2, batch_size=4, random_state=0)
    online.fit(x)
    print("mini-batch steps:", online.n_steps_)
