"""All five estimators coexist and execute through the public family package."""
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler, PolynomialCountSketch, Nystroem
from numojo import zeros, Shape
from std.testing import assert_equal


def contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(4, 2))
    for i in range(x.size):
        x.unsafe_set(i, Scalar[dtype](Float64(i + 1) / 8))
    var rbf = RBFSampler[dtype](random_state=42, n_components=7)
    var skewed = SkewedChi2Sampler[dtype](random_state=42, n_components=7)
    var additive = AdditiveChi2Sampler[dtype]()
    var polynomial = PolynomialCountSketch[dtype](random_state=42, n_components=7)
    var nystroem = Nystroem[dtype](random_state=42, n_components=3)
    assert_equal(rbf.fit_transform(x).shape[1], 7)
    assert_equal(skewed.fit_transform(x).shape[1], 7)
    assert_equal(additive.fit_transform(x).shape[1], 6)
    assert_equal(polynomial.fit_transform(x).shape[1], 7)
    assert_equal(nystroem.fit_transform(x).shape[1], 3)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    var laplacian = Nystroem[DType.float32, output_dtype=DType.float64]("laplacian", random_state=42, n_components=2)
    var x = zeros[DType.float32](Shape(2, 2))
    assert_equal(laplacian.fit_transform(x).shape[1], 2)
    print("PASS all five public estimators, both input dtypes")
