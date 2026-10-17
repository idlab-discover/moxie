"""Native contracts for Nystroem: lifecycle, params, kernels (named/callable/precomputed), errors."""
from numojo import NDArray, Shape, NDArrayStrides, zeros
from moxie.kernel_approximation import Nystroem, NystroemKernelParams, NoCallableKernel
from moxie._kernel_ridge import RowKernel
from moxie.base import clone
from moxie._random import RandomState, entropy_state
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from std.math import exp, isfinite


@fieldwise_init
struct Linear(Copyable, Movable, RowKernel):
    """Callable kernel with its own parameter: scale * <x, y>."""

    var scale: Float64

    def evaluate[dtype: DType](self, X: NDArray[dtype], i: Int, Y: NDArray[dtype], j: Int) raises -> SIMD[dtype, 1]:
        var total = Float64(0)
        for k in range(X.shape[1]):
            total += Float64(X.unsafe_get(i * Int(X.strides[0]) + k * Int(X.strides[1]))) * Float64(Y.unsafe_get(j * Int(Y.strides[0]) + k * Int(Y.strides[1])))
        return SIMD[dtype, 1](self.scale * total)


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def gram_matches[dtype: DType](z: NDArray[dtype], expected: NDArray[dtype], tol: Float64) raises:
    var n = z.shape[0]
    for i in range(n):
        for j in range(n):
            var total = Float64(0)
            for c in range(z.shape[1]):
                total += Float64(z.unsafe_get(i * z.shape[1] + c)) * Float64(z.unsafe_get(j * z.shape[1] + c))
            assert_almost_equal(total, Float64(expected.unsafe_get(i * n + j)), atol=tol)


def contracts[dtype: DType]() raises:
    comptime tol = 1e-3 if dtype == DType.float32 else 1e-9
    var x = zeros[dtype](Shape(6, 2))
    for k in range(12):
        x.unsafe_set(k, Scalar[dtype](Float64((k * 7) % 11) / 10.0))
    var defaults = Nystroem[dtype]()
    assert_equal(defaults.get_params().kernel, "rbf")
    assert_equal(defaults.get_params().n_components, 100)
    assert_false(Bool(defaults.get_params().gamma))
    with assert_raises(contains="random_state=None"):
        defaults.fit(x)
    var model = Nystroem[dtype](gamma=0.5, n_components=4, random_state=0)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(x)
    var z = model.fit_transform(x, zeros[dtype](Shape(6)))
    assert_equal(z.shape[0], 6)
    assert_equal(z.shape[1], 4)
    assert_equal(model.components_.shape[0], 4)
    assert_equal(model.components_.shape[1], 2)
    assert_equal(model.normalization_.shape[0], 4)
    assert_equal(model.component_indices_.size, 4)
    assert_equal(len(model.fit_warnings), 0)
    # Components are copies of the indexed training rows, from permutation(n)[:k].
    var rng = RandomState(0)
    var permutation = rng.permutation(6)
    for r in range(4):
        assert_equal(model.component_indices_.unsafe_get(r), permutation.unsafe_get(r))
        for c in range(2):
            assert_equal(model.components_.unsafe_get(r * 2 + c), x.unsafe_get(Int(permutation.unsafe_get(r)) * 2 + c))
    # Full basis: the map reproduces the exact kernel on the training data.
    var full = Nystroem[dtype]("linear", n_components=10, random_state=1)
    var zf = full.fit_transform(x)
    assert_equal(zf.shape[1], 6)
    assert_equal(len(full.fit_warnings), 1)  # n_components > n_samples
    var rbf_full = Nystroem[dtype](gamma=0.5, n_components=6, random_state=1)
    var zr = rbf_full.fit_transform(x)
    var exact = zeros[dtype](Shape(6, 6))
    for i in range(6):
        for j in range(6):
            var d = 0.0
            for c in range(2):
                d += (Float64(x.unsafe_get(i * 2 + c)) - Float64(x.unsafe_get(j * 2 + c))) ** 2
            exact.unsafe_set(i * 6 + j, Scalar[dtype](exp(-0.5 * d)))
    gram_matches(zr, exact, tol)
    # Determinism, repeat fit, output/learned-state independence, clone.
    same(z, model.transform(x))
    assert_true(model.fit(x).__sklearn_is_fitted__())
    same(z, model.transform(x))
    z.unsafe_set(0, 123)
    assert_true(model.transform(x).unsafe_get(0) != 123)
    var c = clone(model)
    assert_false(c.__sklearn_is_fitted__())
    assert_equal(c.get_params().gamma.value(), 0.5)
    c.fit(x)
    same(c.normalization_, model.normalization_)
    c.components_.unsafe_set(0, 99)
    assert_true(model.components_.unsafe_get(0) != 99)
    var context = RandomState(0)
    var ctx_model = Nystroem[dtype](gamma=0.5, n_components=4)
    ctx_model.fit_with_rng(x, context)
    same(ctx_model.normalization_, model.normalization_)
    var entropy = entropy_state()
    _ = ctx_model.fit_transform_with_rng(x, entropy)
    # Every named kernel fits and transforms.
    for kernel in [String("additive_chi2"), "chi2", "linear", "poly", "polynomial", "rbf", "sigmoid", "cosine"]:
        var m = Nystroem[dtype](kernel, n_components=3, random_state=2)
        assert_equal(m.fit_transform(x).shape[1], 3)
    var laplacian = Nystroem[dtype, NoCallableKernel, DType.float64]("laplacian", n_components=3, random_state=2)
    var promoted: NDArray[DType.float64] = laplacian.fit_transform(x)
    assert_equal(promoted.shape[1], 3)
    comptime if dtype == DType.float32:
        var unpromoted = Nystroem[dtype]("laplacian", random_state=2)
        with assert_raises(contains="output_dtype"):
            unpromoted.fit(x)
    var cloned_laplacian = clone(laplacian)
    assert_false(cloned_laplacian.__sklearn_is_fitted__())
    same(promoted, cloned_laplacian.fit_transform(x))
    comptime if dtype == DType.float32:
        laplacian.set_params(n_jobs=2)
        with assert_raises(contains="output_dtype"):
            _ = laplacian.transform(x)
        var invalid_promotion = Nystroem[dtype, output_dtype=DType.float64](random_state=0)
        with assert_raises(contains="output_dtype"):
            invalid_promotion.fit(x)
    var parallel_laplacian = Nystroem[dtype]("laplacian", n_components=3, random_state=2, n_jobs=2)
    var unpromoted_parallel: NDArray[dtype] = parallel_laplacian.fit_transform(x)
    assert_equal(unpromoted_parallel.shape[1], 3)
    for jobs in [1, 2, -1, -2, 100]:
        var parallel = Nystroem[dtype](gamma=0.5, n_components=4, random_state=0, n_jobs=jobs)
        var parallel_z = parallel.fit_transform(x)
        for i in range(parallel_z.size):
            assert_almost_equal(Float64(parallel_z.unsafe_get(i)), Float64(model.transform(x).unsafe_get(i)), atol=tol)
    var invalid_jobs = Nystroem[dtype](random_state=0, n_jobs=0)
    with assert_raises(contains="n_jobs"):
        invalid_jobs.fit(x)
    model.set_params(n_jobs=2)
    assert_equal(model.get_params().n_jobs.value(), 2)
    model.set_params(n_jobs=1)
    # kernel_params supplies defaults; explicit values override them.
    var kp = Nystroem[dtype]("poly", kernel_params=NystroemKernelParams(Optional[Float64](0.5), Optional[Float64](2.0), Optional[Float64](2.0)), n_components=3, random_state=2)
    var ke = Nystroem[dtype]("poly", gamma=0.5, coef0=2.0, degree=2.0, n_components=3, random_state=2)
    same(kp.fit_transform(x), ke.fit_transform(x))
    var ko = Nystroem[dtype]("poly", gamma=0.5, coef0=2.0, degree=2.0, kernel_params=NystroemKernelParams(Optional[Float64](9.0), None, None), n_components=3, random_state=2)
    same(ko.fit_transform(x), ke.fit_transform(x))
    # Callable kernel: typed struct with its own parameter.
    var callable = Nystroem[dtype, Linear](Linear(2.0), n_components=6, random_state=3)
    var zc = callable.fit_transform(x)
    var linear_exact = zeros[dtype](Shape(6, 6))
    for i in range(6):
        for j in range(6):
            var dot = 0.0
            for k in range(2):
                dot += Float64(x.unsafe_get(i * 2 + k)) * Float64(x.unsafe_get(j * 2 + k))
            linear_exact.unsafe_set(i * 6 + j, Scalar[dtype](2.0 * dot))
    gram_matches(zc, linear_exact, tol * 10)
    var cc = clone(callable)
    assert_equal(cc.get_params().kernel_function.value().scale, 2.0)
    var callable_gamma = callable.get_params()
    callable_gamma.gamma = 1.0
    callable.set_params(callable_gamma)
    with assert_raises(contains="Don't pass gamma"):
        callable.fit(x)
    var parallel_callable = Nystroem[dtype, Linear](Linear(2.0), n_components=6, random_state=3, n_jobs=2)
    gram_matches(parallel_callable.fit_transform(x), linear_exact, tol * 10)
    # Precomputed: basis must be the full training set (reference shape error otherwise).
    var pre = Nystroem[dtype]("precomputed", n_components=6, random_state=4)
    var zp = pre.fit_transform(linear_exact)
    gram_matches(zp, linear_exact, tol * 10)
    var pre_small = Nystroem[dtype]("precomputed", n_components=3, random_state=4)
    with assert_raises(contains="precomputed"):
        pre_small.fit(linear_exact)
    var pre_gamma = Nystroem[dtype]("precomputed", gamma=0.1, n_components=6, random_state=4)
    with assert_raises(contains="Don't pass gamma"):
        pre_gamma.fit(linear_exact)
    # Names and parameter errors.
    var names = model.get_feature_names_out()
    assert_equal(len(names), 4)
    assert_equal(names[3], "nystroem3")
    var bad_kernel = Nystroem[dtype]("gaussian", random_state=0)
    with assert_raises(contains="unknown kernel"):
        bad_kernel.fit(x)
    var bad_gamma = Nystroem[dtype](gamma=-1.0, random_state=0)
    with assert_raises(contains="gamma"):
        bad_gamma.fit(x)
    var bad_degree = Nystroem[dtype]("poly", degree=0.5, random_state=0)
    with assert_raises(contains="degree"):
        bad_degree.fit(x)
    var bad_width = Nystroem[dtype](n_components=0, random_state=0)
    with assert_raises(contains="n_components"):
        bad_width.fit(x)
    var negative = x.copy()
    negative.unsafe_set(0, -1)
    var chi = Nystroem[dtype]("chi2", n_components=6, random_state=0)
    with assert_raises(contains="nonnegative"):
        chi.fit(negative)
    var nonfinite = x.copy()
    nonfinite.unsafe_set(0, Scalar[dtype](1) / Scalar[dtype](0))
    with assert_raises():
        model.fit(nonfinite)
    with assert_raises():
        _ = model.transform(zeros[dtype](Shape(2, 3)))
    with assert_raises():
        model.fit(zeros[dtype](Shape(0, 2)))
    var parallel_chi = Nystroem[dtype]("chi2", n_components=6, random_state=0, n_jobs=2)
    parallel_chi.fit(x)
    var saved = parallel_chi.normalization_.copy()
    with assert_raises(contains="nonnegative"):
        parallel_chi.fit(negative)
    same(saved, parallel_chi.normalization_)
    with assert_raises(contains="nonnegative"):
        _ = parallel_chi.transform(negative)
    # Rank-deficient basis (duplicate rows) still yields finite features.
    var dup = zeros[dtype](Shape(4, 2))
    for k in range(8):
        dup.unsafe_set(k, Scalar[dtype](Float64(k % 2) + 1))
    var dup_model = Nystroem[dtype]("linear", n_components=4, random_state=0)
    var zd = dup_model.fit_transform(dup)
    for k in range(zd.size):
        assert_true(isfinite(zd.unsafe_get(k)))
    # Strided views follow logical order.
    var backing = zeros[dtype](Shape(6, 4))
    for k in range(24):
        backing.unsafe_set(k, Scalar[dtype](Float64(k % 7) / 7.0))
    var strided = backing.view_with_layout(Shape(6, 2), NDArrayStrides(4, -1), 2)
    var dense = zeros[dtype](Shape(6, 2))
    for i in range(6):
        for j in range(2):
            dense.unsafe_set(i * 2 + j, strided.unsafe_get(i * 4 - j))
    var sm = Nystroem[dtype](n_components=3, random_state=5)
    sm.fit(dense)
    same(sm.transform(strided), sm.transform(dense))
    sm.set_params(n_jobs=2)
    same(sm.transform(strided), sm.transform(dense))
    sm.set_params(n_jobs=1)
    var old = sm.normalization_.copy()
    sm.fit(strided)
    same(old, sm.normalization_)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS native lifecycle, params, named/callable/precomputed kernels, errors, rank deficiency and strided contracts")
