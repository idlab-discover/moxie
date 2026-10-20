import numojo as nm
from numojo.core.layout import NDArrayStrides
from moxie.neighbors import NeighborhoodComponentsAnalysis
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone
from moxie.neighbors import NCACallback
from moxie.neighbors._nca_math import Matrix
from moxie._random import RandomState


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 2))
    X.unsafe_set(0, 0)
    X.unsafe_set(1, 0)
    X.unsafe_set(2, 0.2)
    X.unsafe_set(3, 1)
    X.unsafe_set(4, 2)
    X.unsafe_set(5, 0)
    X.unsafe_set(6, 2.4)
    X.unsafe_set(7, 1)
    var y = nm.zeros[DType.float64](nm.Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    var model = NeighborhoodComponentsAnalysis()
    with assert_raises():
        _ = model.transform(X)
    model.fit(X, y)
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.components_.shape[0], 2)
    assert_true(model.n_iter_ > 1)
    var output = model.transform(X)
    assert_equal(output.shape[0], 4)
    var unfitted = clone(model)
    assert_true(not unfitted.__sklearn_is_fitted__())
    for mode in ["identity", "random", "pca", "lda", "auto"]:
        var reduced = NeighborhoodComponentsAnalysis(
            1, init=mode, random_state=42
        )
        reduced.fit(X, y)
        assert_equal(reduced.components_.shape[0], 1)
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS: NCA")


struct Recorder(NCACallback):
    var calls: Int
    var last: Int

    def __init__(out self):
        self.calls = 0
        self.last = 0

    def __call__(mut self, transformation: Matrix, n_iter: Int) raises:
        assert_equal(transformation.ndim, 1)
        assert_equal(n_iter, self.calls + 1)
        self.calls += 1
        self.last = n_iter
        var independent = transformation.copy()
        independent.unsafe_set(0, 999)


def contracts[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 3))
    for i in range(x.size):
        x.unsafe_set(i, (Float64(i % 7) * 0.3).cast[dtype]())
    var y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i % 2).cast[dtype]())
    var model = NeighborhoodComponentsAnalysis[dtype]()
    model.set_params(n_components=2, random_state=42)
    var reset = model.get_params()
    reset.n_components = None
    reset.random_state = None
    model.set_params(reset)
    var params = model.get_params()
    assert_true(not params.n_components)
    assert_equal(params.init, "auto")
    assert_equal(params.max_iter, 50)
    assert_equal(params.tol, 1e-5)
    assert_true(not params.warm_start)
    assert_true(not params.random_state)
    assert_equal(params.verbose, 0)
    with assert_raises():
        _ = model.get_feature_names_out()
    with assert_raises():
        _ = model.fit_transform(x)
    model.fit(x, y)
    var original = model.components_.copy()
    var names = model.get_feature_names_out()
    assert_equal(len(names), 3)
    assert_equal(names[0], "neighborhoodcomponentsanalysis0")
    var wrong_names: List[String] = ["a"]
    with assert_raises():
        _ = model.get_feature_names_out(wrong_names.copy())
    var wrong = nm.zeros[dtype](nm.Shape(6, 2))
    with assert_raises():
        _ = model.transform(wrong)
    var bad_y = nm.zeros[dtype](nm.Shape(5))
    with assert_raises():
        model.fit(x, bad_y)
    var multi = nm.zeros[dtype](nm.Shape(6, 2))
    with assert_raises():
        model.fit(x, multi)
    var fractional = y.copy()
    fractional.unsafe_set(0, 0.5)
    with assert_raises():
        model.fit(x, fractional)
    var nonfinite = x.copy()
    nonfinite.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises():
        model.fit(nonfinite, y)
    with assert_raises():
        _ = model.transform(nonfinite)
    for i in range(original.size):
        assert_equal(original.unsafe_get(i), model.components_.unsafe_get(i))
    model.fit(x, y)
    for i in range(original.size):
        assert_equal(original.unsafe_get(i), model.components_.unsafe_get(i))
    for k in [0, 4]:
        var bad = NeighborhoodComponentsAnalysis[dtype](k)
        with assert_raises():
            bad.fit(x, y)
    for mode in ["invalid", "array"]:
        var bad = NeighborhoodComponentsAnalysis[dtype](init=mode)
        with assert_raises():
            bad.fit(x, y)
    for tolerance in [Float64(-1), Float64("nan"), Float64("-inf")]:
        var bad = NeighborhoodComponentsAnalysis[dtype](tol=tolerance)
        with assert_raises():
            bad.fit(x, y)
    var infinite_tol = NeighborhoodComponentsAnalysis[dtype](tol=Float64("inf"))
    infinite_tol.fit(x, y)
    assert_equal(infinite_tol.n_iter_, 1)
    var bad = NeighborhoodComponentsAnalysis[dtype](max_iter=0)
    with assert_raises():
        bad.fit(x, y)
    bad.set_params(max_iter=1, verbose=-1)
    with assert_raises():
        bad.fit(x, y)
    var bad_seed = NeighborhoodComponentsAnalysis[dtype](random_state=-1)
    with assert_raises():
        bad_seed.fit(x, y)
    var initial = nm.zeros[DType.float64](nm.Shape(2, 3))
    initial.unsafe_set(0, 1)
    initial.unsafe_set(4, 1)
    var explicit = NeighborhoodComponentsAnalysis[dtype](init=initial)
    initial.unsafe_set(0, 99)
    assert_equal(explicit.get_params().init_matrix.value().unsafe_get(0), 1)
    var detached = explicit.get_params()
    detached.init_matrix.value().unsafe_set(0, 77)
    assert_equal(explicit.get_params().init_matrix.value().unsafe_get(0), 1)
    explicit.fit(x, y)
    assert_equal(explicit.components_.shape[0], 2)
    var cloned = clone(explicit)
    assert_true(not cloned.__sklearn_is_fitted__())
    assert_true(not cloned.random_state_)
    assert_equal(cloned.n_iter_, 0)
    var bad_init = nm.zeros[DType.float64](nm.Shape(4, 3))
    explicit.set_params(init=bad_init)
    with assert_raises():
        explicit.fit(x, y)
    var wrong_init = nm.zeros[DType.float64](nm.Shape(2, 2))
    explicit.set_params(init=wrong_init)
    with assert_raises():
        explicit.fit(x, y)
    var mismatch = NeighborhoodComponentsAnalysis[dtype](1, init=initial)
    with assert_raises():
        mismatch.fit(x, y)
    var warm = NeighborhoodComponentsAnalysis[dtype](
        init="identity", warm_start=True, max_iter=1
    )
    warm.fit(x, y)
    var before = warm.components_.copy()
    warm.set_params(tol=1e10, n_components=1)
    warm.fit(x, y)
    # warm_start keeps the previous output dimension even after n_components changes.
    assert_equal(warm.components_.shape[0], 3)
    for i in range(before.size):
        assert_equal(warm.components_.unsafe_get(i), before.unsafe_get(i))
    with assert_raises():
        warm.fit(wrong, y)
    var random = NeighborhoodComponentsAnalysis[dtype](
        init="random", random_state=42, tol=1e10
    )
    random.fit(x, y)
    var rng = RandomState(42)
    for i in range(random.components_.size):
        assert_equal(random.components_.unsafe_get(i), rng.standard_normal())
    random.fit(x, y)
    var fresh = RandomState(42)
    for i in range(random.components_.size):
        assert_equal(random.components_.unsafe_get(i), fresh.standard_normal())
    random.fit(x, y, random_state=rng)
    for i in range(random.components_.size):
        assert_equal(random.components_.unsafe_get(i), fresh.standard_normal())
    var with_callback = NeighborhoodComponentsAnalysis[dtype, Recorder](
        max_iter=3, tol=0
    )
    with_callback.fit(x, y)
    assert_equal(
        with_callback.get_params().callback.calls, with_callback.n_iter_ - 1
    )
    assert_true(with_callback.n_iter_ > 1)
    assert_true(with_callback.components_.unsafe_get(0) != 999)
    # Strided views preserve row order and never alias fitted state.
    var sliced = x.view_with_layout(nm.Shape(3, 3), NDArrayStrides(6, 1), 0)
    var labels = nm.zeros[dtype](nm.Shape(3))
    labels.unsafe_set(1, 1)
    var strided = NeighborhoodComponentsAnalysis[dtype](init="identity")
    strided.fit(sliced, labels)
    var contiguous = sliced.contiguous()
    var independent = NeighborhoodComponentsAnalysis[dtype](init="identity")
    independent.fit(contiguous, labels)
    for i in range(strided.components_.size):
        assert_equal(
            strided.components_.unsafe_get(i),
            independent.components_.unsafe_get(i),
        )
    model.set_params(init="identity")
    var same = nm.zeros[dtype](nm.Shape(6))
    model.fit(x, same)
    assert_equal(model.n_iter_, 1)
    for i in range(9):
        assert_equal(
            model.components_.unsafe_get(i), Float64(1 if i % 4 == 0 else 0)
        )
    for i in range(6):
        same.unsafe_set(i, Float64(i).cast[dtype]())
    model.fit(x, same)
    assert_equal(model.n_iter_, 1)
    var constant = nm.ones[dtype](nm.Shape(6, 3))
    var lda = NeighborhoodComponentsAnalysis[dtype](1, init="lda")
    with assert_raises():
        lda.fit(constant, y)
    var tiny = nm.zeros[dtype](nm.Shape(1, 3))
    var tiny_y = nm.zeros[dtype](nm.Shape(1))
    with assert_raises():
        model.fit(tiny, tiny_y)

    var degenerate = nm.zeros[dtype](nm.Shape(4, 2))
    degenerate.unsafe_set(3, 1)
    degenerate.unsafe_set(4, 2)
    degenerate.unsafe_set(6, 2)
    degenerate.unsafe_set(7, 1)
    var dy = nm.zeros[dtype](nm.Shape(4))
    dy.unsafe_set(2, 1)
    dy.unsafe_set(3, 1)
    var zero_axes = NeighborhoodComponentsAnalysis[dtype](1, init="lda")
    with assert_raises():
        zero_axes.fit(degenerate, dy)
    var two = nm.zeros[dtype](nm.Shape(2, 3))
    var two_y = nm.zeros[dtype](nm.Shape(2))
    var too_many_pcs = NeighborhoodComponentsAnalysis[dtype](3, init="pca")
    with assert_raises():
        too_many_pcs.fit(two, two_y)
