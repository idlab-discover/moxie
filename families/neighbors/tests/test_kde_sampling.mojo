import numojo as nm
from std.math import isfinite
from std.sys import argv
from std.testing import assert_equal, assert_true, assert_raises
from moxie._random import RandomState, seed_global_random_state
from moxie.base import clone
from moxie.neighbors import KernelDensity


def exercise[dtype: DType](dim: Int, seed: Int, weighted: Bool, bandwidth: Float64) raises:
    var backing = nm.zeros[dtype](nm.Shape(8, dim * 2))
    var x = backing.view_with_layout(nm.Shape(4, dim), nm.NDArrayStrides(dim * 4, 2), 0)
    for i in range(4):
        for j in range(dim):
            x.unsafe_set(i * dim * 4 + j * 2, Float64(i * 10 + j % 3).cast[dtype]())
    var weights = nm.zeros[dtype](nm.Shape(4))
    weights.unsafe_set(1, 1)
    weights.unsafe_set(3, 3)
    var model = KernelDensity[dtype](kernel="tophat", bandwidth=bandwidth)
    with assert_raises():
        _ = model.sample(1, seed)
    if weighted:
        model.fit(x, sample_weight=weights.view())
    else:
        model.fit(x)
    var samples = model.sample(7, seed)
    var repeated = model.sample(7, seed)
    assert_equal(samples.shape[0], 7)
    assert_equal(samples.shape[1], dim)
    # Fit owns the strided data; later caller mutation must not change draws.
    x.unsafe_set(0, 999)
    var snapshot = model.sample(7, seed)
    for i in range(7 * dim):
        assert_true(isfinite(samples.unsafe_get(i)))
        assert_equal(samples.unsafe_get(i), repeated.unsafe_get(i))
        assert_equal(samples.unsafe_get(i), snapshot.unsafe_get(i))
        print("S", samples.unsafe_get(i))
    var rng = RandomState(seed)
    var first = model.sample_with_rng(7, rng)
    var second = model.sample_with_rng(3, rng)
    for i in range(7 * dim):
        assert_equal(first.unsafe_get(i), samples.unsafe_get(i))
    for i in range(3 * dim):
        print("N", second.unsafe_get(i))
    print("U", rng.uniform())
    var fresh = clone(model)
    with assert_raises():
        _ = fresh.sample(1, seed)
    with assert_raises():
        _ = model.sample(0, seed)
    with assert_raises():
        _ = model.sample(-1, seed)
    seed_global_random_state(seed)
    var default_first = model.sample(7)
    var default_second = model.sample(3)
    var reference_rng = RandomState(seed)
    var reference_first = model.sample_with_rng(7,reference_rng)
    var reference_second = model.sample_with_rng(3,reference_rng)
    for i in range(7*dim): assert_equal(default_first.unsafe_get(i),reference_first.unsafe_get(i))
    for i in range(3*dim): assert_equal(default_second.unsafe_get(i),reference_second.unsafe_get(i))
    for kernel in [String("epanechnikov"), "exponential", "linear", "cosine"]:
        var params = model.get_params()
        params.kernel = kernel
        model.set_params(params)
        with assert_raises():
            _ = model.sample(1, seed)
    var params = model.get_params()
    params.kernel = "tophat"
    params.bandwidth = bandwidth * 2
    model.set_params(params)
    var zero = nm.zeros[dtype](nm.Shape(1, dim))
    model.fit(zero)
    var single = model.sample(1, seed)
    assert_equal(single.shape[0], 1)
    var norm = Float64(0)
    for j in range(dim):
        norm += single.unsafe_get(j) * single.unsafe_get(j)
    assert_true(norm <= 4 * bandwidth * bandwidth * (1 + 1e-9))


def main() raises:
    var args = argv()
    if len(args) == 1:
        exercise[DType.float64](3, 42, True, 0.75)
    elif args[1] == "f32":
        exercise[DType.float32](Int(args[2]), Int(args[3]), args[4] == "weighted", Float64(args[5]))
    else:
        exercise[DType.float64](Int(args[2]), Int(args[3]), args[4] == "weighted", Float64(args[5]))
