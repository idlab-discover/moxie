"""Native contracts for AdditiveChi2Sampler: lifecycle, parameters, domain, layout, names."""
from numojo import NDArray, Shape, NDArrayStrides, zeros
from moxie.kernel_approximation import AdditiveChi2Sampler
from moxie.base import clone
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from std.math import sqrt, log, cos, sin, cosh


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.shape[0], b.shape[0])
    assert_equal(a.shape[1], b.shape[1])
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def matrix[dtype: DType](rows: Int, cols: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(rows, cols))
    for k in range(x.size):
        x.unsafe_set(k, Scalar[dtype](Float64(k % 5) / 4.0))  # includes zeros
    return x^


def contracts[dtype: DType]() raises:
    comptime tol = 1e-6 if dtype == DType.float32 else 1e-14
    var x = matrix[dtype](3, 2)
    var model = AdditiveChi2Sampler[dtype]()
    assert_equal(model.get_params().sample_steps, 2)
    assert_false(Bool(model.get_params().sample_interval))
    assert_false(model.__sklearn_is_fitted__())
    # Stateless: transform before fit, with the reference layout.
    var z = model.transform(x)
    assert_equal(z.shape[0], 3)
    assert_equal(z.shape[1], 6)
    var interval = 0.5
    for i in range(3):
        for f in range(2):
            var v = Float64(x.unsafe_get(i * 2 + f))
            if v == 0:
                for block in range(3):
                    assert_equal(z.unsafe_get(i * 6 + block * 2 + f), 0)
                continue
            var factor = sqrt(2 * v * interval / cosh(3.14159265358979323846 * interval))
            assert_almost_equal(Float64(z.unsafe_get(i * 6 + f)), sqrt(v * interval), atol=tol)
            assert_almost_equal(Float64(z.unsafe_get(i * 6 + 2 + f)), factor * cos(interval * log(v)), atol=tol)
            assert_almost_equal(Float64(z.unsafe_get(i * 6 + 4 + f)), factor * sin(interval * log(v)), atol=tol)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    # fit records the feature count; transform then checks it.
    var target = zeros[dtype](Shape(3))
    var fitted = model.fit_transform(x, target.copy())
    same(fitted, z)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.n_features_in_, 2)
    var wide = matrix[dtype](3, 3)
    with assert_raises(contains="feature count"):
        _ = model.transform(wide)
    var unfitted_other = AdditiveChi2Sampler[dtype]()
    assert_equal(unfitted_other.transform(wide).shape[1], 9)
    model.fit(wide)
    assert_equal(model.n_features_in_, 3)
    assert_true(model.fit(x).__sklearn_is_fitted__())
    # Output buffers are independent of the input and of each other.
    fitted.unsafe_set(1, 42)
    assert_true(model.transform(x).unsafe_get(1) != 42)
    # Feature names follow the sqrt, cos1, sin1 block order.
    var names = model.get_feature_names_out()
    assert_equal(len(names), 6)
    assert_equal(names[0], "additivechi2sampler_x0_sqrt")
    assert_equal(names[1], "additivechi2sampler_x1_sqrt")
    assert_equal(names[2], "additivechi2sampler_x0_cos1")
    assert_equal(names[4], "additivechi2sampler_x0_sin1")
    var given: List[String] = ["a", "b"]
    assert_equal(model.get_feature_names_out(given.copy())[5], "additivechi2sampler_b_sin1")
    var short: List[String] = ["a"]
    with assert_raises(contains="input_features"):
        _ = model.get_feature_names_out(short.copy())
    # Nonnegative, finite, rank-2, nonempty domain on fit and transform.
    var negative = matrix[dtype](2, 2)
    negative.unsafe_set(3, -1e-3)
    with assert_raises(contains="nonnegative"):
        model.fit(negative)
    with assert_raises(contains="nonnegative"):
        _ = model.transform(negative)
    with assert_raises(contains="nonnegative"):
        _ = AdditiveChi2Sampler[dtype]().transform(negative)
    var nonfinite = matrix[dtype](2, 2)
    nonfinite.unsafe_set(0, Scalar[dtype](1) / Scalar[dtype](0))
    with assert_raises(contains="finite"):
        model.fit(nonfinite)
    with assert_raises():
        model.fit(zeros[dtype](Shape(0, 2)))
    with assert_raises():
        model.fit(zeros[dtype](Shape(2)))
    # Default intervals for 1..3 steps; others require an explicit interval.
    for steps in range(1, 4):
        var m = AdditiveChi2Sampler[dtype](sample_steps=steps)
        m.fit(x)
        assert_equal(m.transform(x).shape[1], 2 * (2 * steps - 1))
        assert_equal(len(m.get_feature_names_out()), 2 * (2 * steps - 1))
    var four = AdditiveChi2Sampler[dtype](sample_steps=4)
    with assert_raises(contains="sample_interval"):
        four.fit(x)
    with assert_raises(contains="sample_interval"):
        _ = four.transform(x)
    four.set_params(sample_interval=0.3)
    four.fit(x)
    assert_equal(four.transform(x).shape[1], 14)
    var zero_steps = AdditiveChi2Sampler[dtype](sample_steps=0)
    with assert_raises(contains="sample_steps"):
        zero_steps.fit(x)
    var negative_interval = AdditiveChi2Sampler[dtype](sample_interval=-0.1)
    with assert_raises(contains="sample_interval"):
        negative_interval.fit(x)
    # Interval zero is allowed by sklearn and maps everything to zero.
    var zero_interval = AdditiveChi2Sampler[dtype](sample_interval=0.0)
    var zeros_out = zero_interval.fit_transform(x)
    for k in range(zeros_out.size):
        assert_equal(zeros_out.unsafe_get(k), 0)
    # Params are copies; clone is unfitted with the same configuration.
    var params = model.get_params()
    params.sample_steps = 3
    assert_equal(model.get_params().sample_steps, 2)
    var c = clone(model)
    assert_false(c.__sklearn_is_fitted__())
    assert_equal(c.n_features_in_, 0)
    same(c.transform(x), model.transform(x))
    model.set_params(params)
    assert_equal(model.transform(x).shape[1], 10)
    assert_equal(c.transform(x).shape[1], 6)
    # Resetting the interval to None needs a full record.
    var reset = four.get_params()
    reset.sample_interval = None
    four.set_params(reset)
    with assert_raises():
        _ = four.transform(x)
    # Negative strides and nonzero offsets follow logical element order.
    var backing = zeros[dtype](Shape(3, 4))
    for k in range(backing.size):
        backing.unsafe_set(k, Scalar[dtype](k))
    var strided = backing.view_with_layout(Shape(2, 2), NDArrayStrides(4, -1), 2)
    var contiguous = zeros[dtype](Shape(2, 2))
    for i in range(2):
        for j in range(2):
            contiguous.unsafe_set(i * 2 + j, strided.unsafe_get(i * 4 - j))
    var plain = AdditiveChi2Sampler[dtype]()
    same(plain.transform(strided), plain.transform(contiguous))
    # Tiny single-entry input.
    var one = zeros[dtype](Shape(1, 1))
    one.unsafe_set(0, 1)
    var single = plain.transform(one)
    assert_almost_equal(Float64(single.unsafe_get(0)), sqrt(0.5), atol=tol)
    assert_almost_equal(Float64(single.unsafe_get(2)), 0.0, atol=tol)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS native stateless transform, fit dims, params, domain, layout, names, clone and strided contracts")
