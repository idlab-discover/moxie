"""Map dense samples with each kernel approximation estimator."""
from numojo import Shape, zeros
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler, PolynomialCountSketch, Nystroem


def main() raises:
    var x = zeros[DType.float64](Shape(4, 2))
    for i in range(x.size):
        x.unsafe_set(i, Float64(i + 1) / 8)
    var rbf = RBFSampler(random_state=42, n_components=8)
    var skewed = SkewedChi2Sampler(random_state=42, n_components=8)
    var additive = AdditiveChi2Sampler()
    var polynomial = PolynomialCountSketch(random_state=42, n_components=8)
    var nystroem = Nystroem(random_state=42, n_components=3)
    print("RBF:", rbf.fit_transform(x))
    print("Skewed chi2:", skewed.fit_transform(x))
    print("Additive chi2:", additive.fit_transform(x))
    print("Polynomial:", polynomial.fit_transform(x))
    print("Nystroem:", nystroem.fit_transform(x))
