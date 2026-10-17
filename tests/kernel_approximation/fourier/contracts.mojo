from numojo import NDArray, Shape, NDArrayStrides, zeros
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler
from moxie.base import clone
from moxie._random import RandomState, entropy_state
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(2, 2))
    for k in range(4):
        x.unsafe_set(k, Scalar[dtype](k))
    var r = RBFSampler[dtype](random_state=42, n_components=7)
    var s = SkewedChi2Sampler[dtype](random_state=42, n_components=7)
    with assert_raises():
        _ = r.transform(x)
    with assert_raises():
        _ = s.transform(x)
    with assert_raises():
        _ = r.get_feature_names_out()
    with assert_raises():
        _ = s.get_feature_names_out()
    var target = zeros[dtype](Shape(2))
    var z = r.fit_transform(x, target.copy())
    var q = s.fit_transform(x, target.copy())
    assert_equal(z.shape[0], 2)
    assert_equal(z.shape[1], 7)
    assert_equal(r.random_weights_.shape[0], 2)
    assert_equal(r.random_offset_.ndim, 1)
    assert_equal(r.n_features_in_, 2)
    assert_equal(s.n_features_in_, 2)
    assert_equal(r.get_feature_names_out()[0], "rbfsampler0")
    assert_equal(s.get_feature_names_out()[6], "skewedchi2sampler6")
    var names: List[String] = ["a", "b"]
    assert_equal(len(r.get_feature_names_out(names.copy())), 7)
    assert_equal(len(s.get_feature_names_out(names.copy())), 7)
    var badnames: List[String] = ["a"]
    with assert_raises():
        _ = r.get_feature_names_out(badnames.copy())
    with assert_raises():
        _ = s.get_feature_names_out(badnames.copy())
    same(z, r.transform(x))
    same(q, s.transform(x))
    assert_true(r.fit(x).__sklearn_is_fitted__())
    assert_true(s.fit(x).__sklearn_is_fitted__())
    same(z, r.transform(x))
    same(q, s.transform(x))
    # Retained arrays and outputs do not alias X or each other.
    z.unsafe_set(0, 123)
    assert_true(r.transform(x).unsafe_get(0) != 123)
    assert_equal(x.unsafe_get(0), 0)
    var weights = r.random_weights_.copy()
    x.unsafe_set(0, 99)
    same(weights, r.random_weights_)
    x.unsafe_set(0, 0)
    var c = clone(r)
    var d = clone(s)
    assert_equal(c.__sklearn_is_fitted__(), False)
    assert_equal(d.__sklearn_is_fitted__(), False)
    c.fit(x)
    d.fit(x)
    same(c.random_weights_, r.random_weights_)
    same(d.random_weights_, s.random_weights_)
    c.random_weights_.unsafe_set(0, 123)
    assert_true(r.random_weights_.unsafe_get(0) != 123)
    var params = r.get_params()
    params.n_components = 5
    assert_equal(r.get_params().n_components, 7)
    assert_true(r.set_params(params).__sklearn_is_fitted__())
    # Learned width remains seven until refit, current width controls scaling.
    assert_equal(r.transform(x).shape[1], 7)
    assert_equal(len(r.get_feature_names_out()), 7)
    r.fit(x)
    assert_equal(r.transform(x).shape[1], 5)
    r.set_params(gamma=0.0, n_components=3, random_state=1)
    r.fit(x)
    for k in range(r.random_weights_.size):
        assert_equal(r.random_weights_.unsafe_get(k), 0)
    s.set_params(skewedness=2.0, n_components=3, random_state=1)
    s.fit(x)
    assert_equal(s.transform(x).shape[1], 3)
    var sp = s.get_params()
    sp.skewedness = -1
    s.set_params(sp)
    s.fit(x)  # Fit does not check the transform domain.
    with assert_raises():
        _ = s.transform(x)
    s.set_params(skewedness=1.0)
    x.unsafe_set(0, -1)
    with assert_raises():
        _ = s.transform(x)
    x.unsafe_set(0, -0.5)
    _ = s.transform(x)
    x.unsafe_set(0, 0)
    var wrong = zeros[dtype](Shape(2, 3))
    with assert_raises():
        _ = r.transform(wrong)
    with assert_raises():
        _ = s.transform(wrong)
    r.fit(wrong)
    s.fit(wrong)
    assert_equal(r.n_features_in_, 3)
    assert_equal(s.n_features_in_, 3)
    var empty = zeros[dtype](Shape(0, 2))
    var vector = zeros[dtype](Shape(2))
    with assert_raises():
        r.fit(empty)
    with assert_raises():
        s.fit(empty)
    with assert_raises():
        r.fit(vector)
    with assert_raises():
        s.fit(vector)
    for v in range(2):
        x.unsafe_set(0, Scalar[dtype](v) / Scalar[dtype](0))
        with assert_raises():
            r.fit(x)
        with assert_raises():
            s.fit(x)
    x.unsafe_set(0, 0)
    r.set_params(n_components=0)
    s.set_params(n_components=-1)
    with assert_raises():
        r.fit(x)
    with assert_raises():
        s.fit(x)
    with assert_raises():
        _ = r.transform(wrong)
    with assert_raises():
        _ = s.transform(wrong)
    r.set_params(n_components=3, gamma=-1.0)
    with assert_raises():
        r.fit(x)
    var invalid = RBFSampler[dtype](gamma=String("auto"), random_state=42)
    with assert_raises():
        invalid.fit(x)
    var invalid_numeric = RBFSampler[dtype](gamma=String("numeric"), random_state=42)
    with assert_raises():
        invalid_numeric.fit(x)
    var huge = RBFSampler[dtype](n_components=Int.MAX, random_state=42)
    var one = zeros[dtype](Shape(1, 1))
    with assert_raises():
        huge.fit(one)
    var invalid_seed = SkewedChi2Sampler[dtype](random_state=-1)
    with assert_raises():
        invalid_seed.fit(x)
    var default_r = RBFSampler[dtype]()
    var default_s = SkewedChi2Sampler[dtype]()
    assert_equal(default_r.get_params().n_components, 100)
    assert_equal(default_s.get_params().skewedness, 1.0)
    with assert_raises():
        default_r.fit(x)
    with assert_raises():
        default_s.fit(x)
    var rng = RandomState(7)
    var snapshot = rng.copy()
    _ = default_r.fit_transform_with_rng(x, rng)
    var first = default_r.random_weights_.copy()
    default_r.fit_with_rng(x, snapshot)
    same(first, default_r.random_weights_)
    default_r.fit_with_rng(x, rng)
    assert_true(first.unsafe_get(0) != default_r.random_weights_.unsafe_get(0))
    var context = entropy_state()
    _ = default_s.fit_transform_with_rng(x, context)
    var seeded = SkewedChi2Sampler[dtype](random_state=7)
    seeded.fit(x)
    var state = RandomState(7)
    default_s.fit_with_rng(x, state)
    same(seeded.random_weights_, default_s.random_weights_)
    # Negative strides and nonzero offset must follow logical element order.
    var backing = zeros[dtype](Shape(3, 4))
    for k in range(backing.size):
        backing.unsafe_set(k, Scalar[dtype](k))
    var strided = backing.view_with_layout(Shape(2, 2), NDArrayStrides(4, -1), 2)
    var contiguous = zeros[dtype](Shape(2, 2))
    for i in range(2):
        for j in range(2):
            contiguous.unsafe_set(i*2+j, strided.unsafe_get(i*4-j))
    var sr = RBFSampler[dtype](gamma=String("scale"), random_state=42)
    var ss = SkewedChi2Sampler[dtype](random_state=42)
    sr.fit(contiguous)
    ss.fit(contiguous)
    same(sr.transform(strided), sr.transform(contiguous))
    same(ss.transform(strided), ss.transform(contiguous))
    var old = sr.random_weights_.copy()
    sr.fit(strided)
    same(old, sr.random_weights_)
    var constant = zeros[dtype](Shape(1, 1))
    sr.fit(constant)
    assert_equal(sr._gamma, 1.0)
    var binary = zeros[dtype](Shape(2, 1))
    binary.unsafe_set(1, 1)
    sr.fit(binary)
    assert_equal(sr._gamma, 4.0)


def gamma_updates[dtype: DType]() raises:
    var x = zeros[dtype](Shape(2, 1))
    x.unsafe_set(1, 1)
    var model = RBFSampler[dtype](gamma=0.5, n_components=7, random_state=42)
    var before = model.fit_transform(x)
    var weights = model.random_weights_.copy()
    var offsets = model.random_offset_.copy()
    assert_true(model.set_params(gamma=String("scale")).__sklearn_is_fitted__())
    assert_equal(model._gamma, 0.5)
    same(weights, model.random_weights_)
    same(offsets, model.random_offset_)
    same(before, model.transform(x))
    var copied = clone(model)
    assert_equal(copied.__sklearn_is_fitted__(), False)
    assert_equal(copied.get_params().gamma_mode, "scale")
    var fresh = RBFSampler[dtype](gamma=String("scale"), n_components=7, random_state=42)
    var expected = fresh.fit_transform(x)
    same(expected, model.fit_transform(x))
    same(expected, copied.fit_transform(x))
    assert_equal(model._gamma, 4.0)
    same(fresh.random_weights_, model.random_weights_)
    same(fresh.random_offset_, model.random_offset_)
    _ = model.set_params(gamma=0.25)
    same(expected, model.transform(x))
    assert_equal(model.get_params().gamma_mode, "numeric")
    var numeric = RBFSampler[dtype](gamma=0.25, n_components=7, random_state=42)
    same(numeric.fit_transform(x), model.fit_transform(x))
    assert_equal(model._gamma, 0.25)
    # The string overload supports combined updates and fluent fitting.
    _ = model.set_params(gamma=String("scale"), n_components=3, random_state=1).fit(x)
    var combined = RBFSampler[dtype](gamma=String("scale"), n_components=3, random_state=1)
    same(combined.fit_transform(x), model.transform(x))
    for invalid in [String("auto"), String("numeric"), String("")]:
        var previous = model.transform(x)
        _ = model.set_params(gamma=invalid)
        same(previous, model.transform(x))
        with assert_raises():
            _ = model.fit(x)
        var invalid_clone = clone(model)
        with assert_raises():
            _ = invalid_clone.fit(x)
        _ = model.set_params(gamma=String("scale"))
        same(previous, model.fit_transform(x))


def main() raises:
    gamma_updates[DType.float32]()
    gamma_updates[DType.float64]()
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS native lifecycle, params, clone, names, domain, dtype, strided and RNG contracts")
