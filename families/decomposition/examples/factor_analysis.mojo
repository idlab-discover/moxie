"""Two-factor model of six correlated measurements with varimax rotation."""
from numojo import Shape, zeros
from moxie.decomposition import FactorAnalysis


def main() raises:
    var x = zeros[DType.float64](Shape(40, 6))
    for i in range(40):
        var a = Float64((i * 37) % 17) / 17 - 0.5   # latent factor 1
        var b = Float64((i * 11) % 13) / 13 - 0.5   # latent factor 2
        for j in range(6):
            var noise = Float64((i * 5 + j * 7) % 19) / 190 - 0.05
            x.unsafe_set(i * 6 + j, (a if j < 3 else b) * Float64(j + 1) + noise)
    var fa = FactorAnalysis(n_components=2, rotation=Optional[String]("varimax"), random_state=0)
    var latent = fa.fit_transform(x)
    print("latent shape:", latent.shape[0], latent.shape[1])
    print("EM iterations:", fa.n_iter_)
    print("noise variances:", fa.noise_variance_.value())
    print("average log-likelihood:", fa.score(x))
