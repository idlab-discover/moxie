import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from std.utils.numerics import inf
from moxie.cluster import KMeans, kmeans_plusplus, k_means
from moxie.cluster._kmeans import KMeansInit, KMeansNInit
from moxie._random import RandomState, seed_global_random_state
from moxie.base import clone


def checks[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    for i in range(6):
        x[nm.Item(i, 0)] = SIMD[dtype, 1](i // 3 * 10)
        x[nm.Item(i, 1)] = SIMD[dtype, 1](i % 3)
    var w = nm.ones[dtype](nm.Shape(6))
    w[nm.Item(0)] = 0
    var a = kmeans_plusplus(x, 2, random_state=42, sample_weight=w.view())
    var b = kmeans_plusplus(x, 2, random_state=42, sample_weight=w.view())
    for i in range(2):
        assert_equal(a.indices.item(i), b.indices.item(i))
        for f in range(2): assert_equal(a.centers.item(i, f), x.item(Int(a.indices.item(i)), f))
    var state = RandomState(42)
    var borrowed = kmeans_plusplus(x, 2, random_state=state, sample_weight=w.view())
    for i in range(2): assert_equal(a.indices.item(i), borrowed.indices.item(i))
    var expected = RandomState(42)
    for _ in range(3): _ = expected.random_sample()
    assert_equal(state.next_uint32(), expected.next_uint32())
    seed_global_random_state(42)
    var global_result = kmeans_plusplus(x, 2, sample_weight=w.view())
    for i in range(2): assert_equal(global_result.indices.item(i), a.indices.item(i))
    var reversed_x = x.view_with_layout(nm.Shape(6, 2), nm.NDArrayStrides(-2, -1), 11)
    var reversed_result = kmeans_plusplus(reversed_x, 3, random_state=12)
    for i in range(3):
        for f in range(2): assert_equal(reversed_result.centers.item(i, f), reversed_x.item(Int(reversed_result.indices.item(i)), f))
    var identical = nm.ones[dtype](nm.Shape(6, 2))
    var duplicates = kmeans_plusplus(identical, 5, random_state=42)
    for i in range(1, 5): assert_equal(duplicates.indices.item(i), Int64(0))
    with assert_raises(): _ = kmeans_plusplus(x, 0)
    with assert_raises(): _ = kmeans_plusplus(x, 7)
    with assert_raises(): _ = kmeans_plusplus(x, 2, n_local_trials=0)
    with assert_raises(): _ = kmeans_plusplus(x, 2, random_state=-1)
    with assert_raises(): _ = kmeans_plusplus(w, 2)
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises(): _ = kmeans_plusplus(empty, 2)
    var wrong = nm.zeros[dtype](nm.Shape(3))
    with assert_raises(): _ = kmeans_plusplus(x, 2, x_squared_norms=wrong.view())
    with assert_raises(): _ = kmeans_plusplus(x, 2, sample_weight=wrong.view())
    var zero = nm.zeros[dtype](nm.Shape(6))
    with assert_raises(): _ = kmeans_plusplus(x, 2, sample_weight=zero.view())
    w[nm.Item(0)] = -1
    with assert_raises(): _ = kmeans_plusplus(x, 2, sample_weight=w.view())
    w[nm.Item(0)] = 0
    var invalid = x.copy()
    invalid[nm.Item(0, 0)] = inf[dtype]()
    with assert_raises(): _ = kmeans_plusplus(invalid, 2)
    for algorithm in [String("lloyd"), String("elkan")]:
        var model = KMeans[dtype](2, random_state=42, algorithm=algorithm)
        with assert_raises(): _ = model.predict(x)
        with assert_raises(): _ = model.transform(x)
        with assert_raises(): _ = model.score(x)
        with assert_raises(): _ = model.fit(invalid)
        with assert_raises(): _ = model.fit(x, sample_weight=wrong.view())
        _ = model.fit(x, sample_weight=w.view())
        with assert_raises(): _ = model.score(x, sample_weight=wrong.view())
        with assert_raises(): _ = model.predict(invalid)
        assert_true(model.__sklearn_is_fitted__())
        assert_equal(model.n_features_in_, 2)
        assert_equal(model.n_distinct_clusters_, 2)
        var labels = model.predict(x)
        var transformed = model.transform(x)
        for i in range(6): assert_equal(labels.item(i), model.labels_.item(i))
        assert_true(abs(model.score(x, sample_weight=w.view()) + model.inertia_) < 1e-5)
        var independent = clone(model)
        assert_false(independent.__sklearn_is_fitted__())
        assert_equal(independent.get_params().n_clusters, 2)
        var old_centers = model.cluster_centers_.copy()
        _ = model.fit(x, sample_weight=w.view())
        for i in range(2):
            for f in range(2): assert_equal(old_centers.item(i, f), model.cluster_centers_.item(i, f))
        var parameters = model.get_params()
        parameters.n_clusters = 1
        _ = model.set_params(parameters)
        _ = model.fit(identical)
        assert_equal(model.cluster_centers_.shape[0], 1)
        assert_equal(model.inertia_, Float64(0))
        with assert_raises(): _ = model.fit(empty)
        assert_true(model.__sklearn_is_fitted__())
        with assert_raises(): _ = model.predict(nm.zeros[dtype](nm.Shape(2, 3)))
        _ = model.set_params(n_clusters=2, max_iter=1)
        var fit_labels = model.fit_predict(x)
        var prediction = model.predict(x)
        for i in range(6): assert_equal(fit_labels.item(i), prediction.item(i))
        var fit_distances = model.fit_transform(x)
        var distances = model.transform(x)
        for i in range(6):
            for j in range(2): assert_equal(fit_distances.item(i, j), distances.item(i, j))
    var explicit_init = KMeansInit[dtype](a.centers)
    var model = KMeans[dtype](2, init=explicit_init^, n_init=3, copy_x=False)
    _ = model.fit(x)
    assert_equal(x.item(0, 0), SIMD[dtype, 1](0))
    var duplicate_model = KMeans[dtype](4, random_state=0)
    _ = duplicate_model.fit(identical)
    assert_equal(duplicate_model.n_distinct_clusters_, 1)
    var helper = k_means(x, 2, random_state=42)
    assert_false(Bool(helper.n_iter))
    var full = k_means(x, 2, random_state=42, return_n_iter=True)
    assert_true(Bool(full.n_iter))
    for i in range(6): assert_equal(helper.labels.item(i), full.labels.item(i))
    var helper_rng = RandomState(42)
    var borrowed_helper = k_means(x, 2, random_state=helper_rng, return_n_iter=True)
    for i in range(6): assert_equal(borrowed_helper.labels.item(i), full.labels.item(i))
    var original_init = a.centers.copy()
    var owned_model = KMeans[dtype](2, init=KMeansInit[dtype](original_init))
    original_init[nm.Item(0, 0)] = 999
    assert_true(owned_model.params.init.centers.value().item(0, 0) != SIMD[dtype, 1](999))
    var owned_clone = clone(owned_model)
    owned_model.params.init.centers.value()[nm.Item(0, 0)] = 888
    assert_true(owned_clone.params.init.centers.value().item(0, 0) != SIMD[dtype, 1](888))
    var settings = model.get_params()
    settings.init = KMeansInit[dtype]("unsupported")
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.init = KMeansInit[dtype]()
    settings.algorithm = "unsupported"
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.algorithm = "lloyd"
    settings.n_init = KMeansNInit(0)
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.n_init = KMeansNInit("bad")
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.n_init = KMeansNInit()
    settings.tol = -1
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.tol = 1e-4
    settings.max_iter = 0
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.max_iter = 300
    settings.n_clusters = 7
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.n_clusters = 0
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)
    settings.n_clusters = 2
    settings.init = KMeansInit[dtype](wrong)
    _ = model.set_params(settings)
    with assert_raises(): _ = model.fit(x)


def main() raises:
    checks[DType.float32]()
    checks[DType.float64]()
    print("PASS KMeans lifecycle, weights, RNG, ownership and invalid inputs")
