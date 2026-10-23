"""FastICA lifecycle, ownership, invalid settings and dtype contracts."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs, sin
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import FastICA, fastica
from moxie.decomposition._fastica import WhitenOption


def mixed[dtype: DType](n: Int) raises -> NDArray[dtype]:
    """Two sources (sine, sawtooth) mixed into three observed channels."""
    var x = zeros[dtype](Shape(n, 3))
    for i in range(n):
        var s1 = sin(Float64(i) * 0.3)
        var s2 = Float64(i % 7) / 3.5 - 1
        x.unsafe_set(i * 3, Scalar[dtype](s1 + 0.5 * s2))
        x.unsafe_set(i * 3 + 1, Scalar[dtype](0.3 * s1 - s2))
        x.unsafe_set(i * 3 + 2, Scalar[dtype](0.8 * s1 + 0.2 * s2 + 0.01 * Float64(i % 3)))
    return x^


def rejects[dtype: DType](var model: FastICA[dtype], x: NDArray[dtype]) raises:
    with assert_raises(): model.fit(x)


def check[dtype: DType]() raises:
    var x = mixed[dtype](60)
    var model = FastICA[dtype](2, random_state=0)
    with assert_raises(): _ = model.transform(x)
    var s = model.fit_transform(x)
    assert_equal(s.shape[0], 60)
    assert_equal(s.shape[1], 2)
    assert_equal(model.components_.value().shape[1], 3)
    assert_equal(model.mixing_.value().shape[0], 3)
    assert_equal(model.whitening_.value().shape[0], 2)
    assert_true(model.n_iter_ >= 1)
    assert_equal(model.get_feature_names_out()[0], "fastica0")
    # unit-variance sources have unit standard deviation.
    for j in range(2):
        var total: Float64 = 0
        var squares: Float64 = 0
        for i in range(60):
            total += Float64(s.unsafe_get(i * 2 + j))
        for i in range(60):
            squares += (Float64(s.unsafe_get(i * 2 + j)) - total / 60) ** 2
        assert_true(abs(squares / 60 - 1) < 1e-3)
    # Seeded refits are identical; state does not alias the input.
    var first = model.components_.value().unsafe_get(0)
    model.fit(x)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x.unsafe_set(0, 99)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x = mixed[dtype](60)
    var restored = model.inverse_transform(model.transform(x))
    assert_equal(restored.shape[1], 3)
    _ = model.transform(x, copy=False)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_params(algorithm="deflation", fun="cube", whiten_solver="eigh")
    copy.fit(x.view_with_layout(Shape(60, 3), NDArrayStrides(-3, 1), 177))
    assert_true(copy.__sklearn_is_fitted__())
    var raw = FastICA[dtype](whiten=WhitenOption(False), random_state=1, max_iter=500)
    raw.fit(x)
    assert_false(Bool(raw.mean_))
    assert_equal(raw.components_.value().shape[0], 3)
    rejects(FastICA[dtype](0), x)
    rejects(FastICA[dtype](2, algorithm="symmetric"), x)
    rejects(FastICA[dtype](2, whiten=WhitenOption(True)), x)
    rejects(FastICA[dtype](2, whiten=WhitenOption(String("pca"))), x)
    rejects(FastICA[dtype](2, fun="tanh"), x)
    rejects(FastICA[dtype](2, max_iter=0), x)
    rejects(FastICA[dtype](2, tol=-1.0), x)
    rejects(FastICA[dtype](2, whiten_solver="qr"), x)
    var alpha = Dict[String, Float64]()
    alpha["alpha"] = 3.0
    rejects(FastICA[dtype](2, fun_args=alpha^), x)
    var bad_w = zeros[DType.float64](Shape(3, 3))
    rejects(FastICA[dtype](2, w_init=bad_w^), x)
    rejects(FastICA[dtype](2), zeros[dtype](Shape(1, 3)))
    with assert_raises(): _ = model.transform(zeros[dtype](Shape(2, 4)))
    with assert_raises(): _ = model.inverse_transform(zeros[dtype](Shape(2, 3)))
    var result = fastica(x, 2, random_state=0)
    assert_equal(result.W.shape[0], 2)
    assert_equal(result.K.value().shape[1], 3)
    # sklearn still returns S for unit-variance whitening (it needs it for the scaling).
    var unit_sources = fastica(x, 2, random_state=0, compute_sources=False)
    assert_true(Bool(unit_sources.S))
    assert_equal(unit_sources.S.value().shape[1], 2)
    var no_sources = fastica(x, 2, random_state=0, compute_sources=False, whiten=WhitenOption(String("arbitrary-variance")))
    assert_false(Bool(no_sources.S))
    var extra = Dict[String, Float64]()
    extra["alpha"] = 1.5
    extra["beta"] = 2.0
    var ignored = FastICA[dtype](2, fun_args=extra^, random_state=0)
    _ = ignored.fit(x)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: FastICA contracts in Float32 and Float64")
