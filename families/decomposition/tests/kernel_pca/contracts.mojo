"""KernelPCA lifecycle, validation, zero spectra, ownership and layout."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from std.utils.numerics import inf
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import KernelPCA


def check[dtype: DType]() raises:
    var x = zeros[dtype](Shape(7, 3))
    for i in range(7):
        for j in range(3): x.unsafe_set(i * 3 + j, Scalar[dtype]((i * 3 + j * j) % 5))
    var m = KernelPCA[dtype](2, kernel="rbf", fit_inverse_transform=True)
    with assert_raises(): _ = m.transform(x)
    with assert_raises(): _ = m.inverse_transform(x)
    var z = m.fit_transform(x)
    assert_equal(z.shape[1], 2)
    assert_equal(m.get_feature_names_out()[1], "kernelpca1")
    assert_true(m.eigenvalues_.value().unsafe_get(0) >= m.eigenvalues_.value().unsafe_get(1))
    var first = m.X_fit_.value().unsafe_get(0)
    x.unsafe_set(0, 100)
    assert_equal(m.X_fit_.value().unsafe_get(0), first)
    x.unsafe_set(0, first)
    var c = clone(m)
    assert_false(c.__sklearn_is_fitted__())
    var params = c.get_params()
    params.kernel = "linear"
    params.n_components = 1
    c.set_params(params)
    c.fit(x)
    assert_equal(c.eigenvalues_.value().size, 1)
    c.set_params(kernel="rbf")
    c.fit(x.view_with_layout(Shape(7, 3), NDArrayStrides(-3, 1), 18))
    assert_equal(c.n_features_in_, 3)
    m.fit(x)
    var again = m.fit_transform(x)
    for i in range(z.size): assert_true(abs(Float64(z.unsafe_get(i) - again.unsafe_get(i))) < 1e-5)
    with assert_raises(): _ = m.transform(zeros[dtype](Shape(2, 4)))
    with assert_raises(): _ = m.inverse_transform(zeros[dtype](Shape(2, 4)))
    var bad = KernelPCA[dtype](0)
    with assert_raises(): bad.fit(x)
    bad = KernelPCA[dtype](2, eigen_solver="invalid")
    with assert_raises(): bad.fit(x)
    bad = KernelPCA[dtype](2, copy_X=False)
    with assert_raises(): bad.fit(x)
    bad = KernelPCA[dtype](2, alpha=-1)
    with assert_raises(): bad.fit(x)
    bad = KernelPCA[dtype](2, kernel="precomputed", fit_inverse_transform=True)
    with assert_raises(): bad.fit(x)
    bad = KernelPCA[dtype](2)
    with assert_raises(): bad.fit(zeros[dtype](Shape(0, 3)))
    x.unsafe_set(0, inf[dtype]())
    with assert_raises(): bad.fit(x)
    var zero = zeros[dtype](Shape(3, 2))
    var all = KernelPCA[dtype]()
    assert_equal(all.fit_transform(zero).shape[1], 0)
    assert_equal(all.transform(zero).shape[1], 0)
    var keep = KernelPCA[dtype](3)
    assert_equal(keep.fit_transform(zero).shape[1], 3)
    var indefinite = zeros[dtype](Shape(3, 3))
    for i in range(3): indefinite.unsafe_set(i * 3 + i, -1)
    var pre = KernelPCA[dtype](kernel="precomputed")
    with assert_raises(): pre.fit(indefinite)
    with assert_raises(): _ = keep.inverse_transform(zero)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: KernelPCA contracts Float32/Float64")
