"""Numerical edges shared by all five maps, mirroring sklearn at 4358542a.

- NaN/inf input: sklearn's validate_data raises ValueError in fit and transform
  (AdditiveChi2Sampler included, even unfitted); no estimator state changes.
- One sample: every map fits; Nystroem clips the basis to one row and warns.
- Constant features: RBF gamma="scale" falls back to gamma 1 (X.var() == 0).
- Fixed seeds are deterministic across instances; refit on a different width
  replaces the learned arrays, and a later refit clears Nystroem's warning.
"""
from numojo import NDArray, Shape, zeros
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler, PolynomialCountSketch, Nystroem
from std.testing import assert_equal, assert_true, assert_false, assert_raises


def matrix[dtype: DType](rows: Int, cols: Int, offset: Float64 = 0.0) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(rows, cols))
    for k in range(rows * cols):
        x.unsafe_set(k, Scalar[dtype](offset + Float64((k * 5) % 7) / 4))
    return x^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def nonfinite[dtype: DType]() raises:
    var good = matrix[dtype](4, 3)
    for bad_value in [Float64.MAX * 2, -Float64.MAX * 2, Float64.MAX * 0 + (Float64.MAX * 2 - Float64.MAX * 2)]:
        var bad = good.copy()
        bad.unsafe_set(4, Scalar[dtype](bad_value))
        var r = RBFSampler[dtype](random_state=0, n_components=5)
        var s = SkewedChi2Sampler[dtype](random_state=0, n_components=5)
        var a = AdditiveChi2Sampler[dtype]()
        var p = PolynomialCountSketch[dtype](random_state=0, n_components=5)
        var n = Nystroem[dtype](random_state=0, n_components=2)
        with assert_raises(contains="ValueError"):
            _ = r.fit(bad)
        with assert_raises(contains="ValueError"):
            _ = s.fit(bad)
        with assert_raises(contains="ValueError"):
            _ = a.fit(bad)
        with assert_raises(contains="ValueError"):
            _ = a.transform(bad)
        with assert_raises(contains="ValueError"):
            _ = p.fit(bad)
        with assert_raises(contains="ValueError"):
            _ = n.fit(bad)
        assert_false(r.__sklearn_is_fitted__() or s.__sklearn_is_fitted__() or a.__sklearn_is_fitted__())
        assert_false(p.__sklearn_is_fitted__() or n.__sklearn_is_fitted__())
        _ = r.fit(good)
        _ = s.fit(good)
        _ = p.fit(good)
        _ = n.fit(good)
        with assert_raises(contains="ValueError"):
            _ = r.transform(bad)
        with assert_raises(contains="ValueError"):
            _ = s.transform(bad)
        with assert_raises(contains="ValueError"):
            _ = p.transform(bad)
        with assert_raises(contains="ValueError"):
            _ = n.transform(bad)


def single_sample[dtype: DType]() raises:
    var x = matrix[dtype](1, 3, 0.5)
    var r = RBFSampler[dtype](random_state=0, n_components=4)
    var s = SkewedChi2Sampler[dtype](random_state=0, n_components=4)
    var a = AdditiveChi2Sampler[dtype](sample_steps=3)
    var p = PolynomialCountSketch[dtype](random_state=0, n_components=4)
    var n = Nystroem[dtype](random_state=0, n_components=4)
    assert_equal(r.fit_transform(x).shape[1], 4)
    assert_equal(s.fit_transform(x).shape[1], 4)
    assert_equal(a.fit_transform(x).shape[1], 15)
    assert_equal(p.fit_transform(x).shape[1], 4)
    var z = n.fit_transform(x)
    assert_equal(z.shape[0], 1)
    assert_equal(z.shape[1], 1)
    assert_equal(n.components_.shape[0], 1)
    assert_equal(len(n.fit_warnings), 1)
    assert_true(n.fit_warnings[0].startswith("n_components > n_samples. This is not possible."))
    assert_equal(len(n.get_feature_names_out()), 1)
    # A refit without clipping clears the recorded warning (sklearn would not warn).
    _ = n.set_params(n_components=1)
    _ = n.fit(x)
    assert_equal(len(n.fit_warnings), 0)


def constant_scale_gamma[dtype: DType]() raises:
    var x = zeros[dtype](Shape(5, 2))
    for k in range(x.size):
        x.unsafe_set(k, Scalar[dtype](3.5))
    var r = RBFSampler[dtype](gamma=String("scale"), random_state=0, n_components=3)
    _ = r.fit(x)
    assert_equal(r._gamma, 1.0)


def determinism_and_refit[dtype: DType]() raises:
    var x = matrix[dtype](6, 3)
    var wide = matrix[dtype](6, 5)
    var r1 = RBFSampler[dtype](random_state=7, n_components=6)
    var r2 = RBFSampler[dtype](random_state=7, n_components=6)
    same(r1.fit_transform(x), r2.fit_transform(x))
    var s1 = SkewedChi2Sampler[dtype](random_state=7, n_components=6)
    var s2 = SkewedChi2Sampler[dtype](random_state=7, n_components=6)
    same(s1.fit_transform(x), s2.fit_transform(x))
    var p1 = PolynomialCountSketch[dtype](random_state=7, n_components=6)
    var p2 = PolynomialCountSketch[dtype](random_state=7, n_components=6)
    same(p1.fit_transform(x), p2.fit_transform(x))
    var n1 = Nystroem[dtype](random_state=7, n_components=4)
    var n2 = Nystroem[dtype](random_state=7, n_components=4)
    same(n1.fit_transform(x), n2.fit_transform(x))
    # Refit on a different width replaces every learned array.
    _ = r1.fit(wide)
    assert_equal(r1.random_weights_.shape[0], 5)
    assert_equal(r1.get_n_features_in(), 5)
    with assert_raises(contains="ValueError"):
        _ = r1.transform(x)
    _ = p1.fit(wide)
    assert_equal(p1.indexHash_.shape[1], 5)
    _ = n1.fit(wide)
    assert_equal(n1.components_.shape[1], 5)
    var a = AdditiveChi2Sampler[dtype]()
    _ = a.fit(x)
    with assert_raises(contains="ValueError"):
        _ = a.transform(wide)
    _ = a.fit(wide)
    assert_equal(a.transform(wide).shape[1], 15)


def main() raises:
    nonfinite[DType.float32]()
    nonfinite[DType.float64]()
    single_sample[DType.float32]()
    single_sample[DType.float64]()
    constant_scale_gamma[DType.float32]()
    constant_scale_gamma[DType.float64]()
    determinism_and_refit[DType.float32]()
    determinism_and_refit[DType.float64]()
    print("PASS kernel map edges: nonfinite input, one sample, constant scale gamma, determinism, refit")
