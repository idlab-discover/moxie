import numojo as nm
from std.utils.numerics import inf
from std.testing import assert_equal, assert_true, assert_raises
from moxie.cluster import BisectingKMeans
from moxie.base import clone
from moxie._random import RandomState, seed_global_random_state


def check[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(9, 2))
    for i in range(9):
        x[nm.Item(i, 0)] = SIMD[dtype, 1]((i // 3) * 10)
        x[nm.Item(i, 1)] = SIMD[dtype, 1](i % 3)
    var m = BisectingKMeans[dtype](3, random_state=42)
    assert_equal(m.params.init, "random")
    assert_equal(m.params.n_init, 1)
    with assert_raises(): var unused = m.predict(x)
    _ = m.fit(x)
    var labels = m.predict(x)
    assert_equal(len(m._nodes), 5)
    for i in range(9): assert_equal(labels.item(i), m.labels_.item(i))
    var old = m.cluster_centers_.copy()
    _ = m.fit(x)
    for i in range(old.size): assert_equal(old.unsafe_get(i), m.cluster_centers_.unsafe_get(i))
    var c = clone(m)
    assert_true(not c.__sklearn_is_fitted__())
    _ = c.set_params(bisecting_strategy=String("largest_cluster"), n_init=3, algorithm=String("elkan"))
    _ = c.fit(x)
    assert_equal(c.cluster_centers_.shape[0], 3)
    var transformed = c.fit_transform(x)
    assert_equal(transformed.shape[1], 3)
    var fp = c.fit_predict(x)
    for i in range(9): assert_equal(fp.item(i), c.labels_.item(i))
    var params = c.get_params()
    params.n_clusters = 1
    _ = c.set_params(params)
    var w = nm.ones[dtype](nm.Shape(9))
    w.unsafe_set(0, 100)
    _ = c.fit(x, sample_weight=w.view())
    assert_equal(c.cluster_centers_.item(0, 0), SIMD[dtype, 1](10))
    assert_equal(c.cluster_centers_.item(0, 1), SIMD[dtype, 1](1))
    var rng = RandomState(42)
    _ = m.fit(x, random_state=rng)
    for i in range(old.size): assert_equal(old.unsafe_get(i), m.cluster_centers_.unsafe_get(i))
    seed_global_random_state(42)
    var global_model = BisectingKMeans[dtype](3)
    _ = global_model.fit(x)
    for i in range(old.size): assert_equal(old.unsafe_get(i), global_model.cluster_centers_.unsafe_get(i))
    var reversed_x = x.view_with_layout(nm.Shape(9, 2), nm.NDArrayStrides(-2, -1), 17)
    var materialized = nm.zeros[dtype](nm.Shape(9, 2))
    for i in range(9):
        for f in range(2): materialized[nm.Item(i, f)] = reversed_x.item(i, f)
    var strided = BisectingKMeans[dtype](3, random_state=42, copy_x=False)
    var dense = BisectingKMeans[dtype](3, random_state=42)
    _ = strided.fit(reversed_x)
    _ = dense.fit(materialized)
    for i in range(9): assert_equal(strided.labels_.item(i), dense.labels_.item(i))
    for i in range(9):
        for f in range(2): assert_equal(reversed_x.item(i, f), materialized.item(i, f))
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises(): _ = m.fit(empty)
    var invalid = x.copy()
    invalid[nm.Item(0, 0)] = inf[dtype]()
    with assert_raises(): _ = m.fit(invalid)
    var wrong_weight = nm.ones[dtype](nm.Shape(1))
    with assert_raises(): _ = m.fit(x, sample_weight=wrong_weight.view())
    with assert_raises(): var unused = m.score(x, sample_weight=wrong_weight.view())
    var bad = nm.zeros[dtype](nm.Shape(3, 1))
    with assert_raises(): var unused = m.predict(bad)
    _ = m.set_params(init=String("array"))
    with assert_raises(): _ = m.fit(x)
    assert_true(m.__sklearn_is_fitted__())
    for i in range(old.size): assert_equal(old.unsafe_get(i), m.cluster_centers_.unsafe_get(i))
    _ = m.set_params(init=String("random"), bisecting_strategy=String("invalid"))
    with assert_raises(): _ = m.fit(x)
    _ = m.set_params(bisecting_strategy=String("biggest_inertia"), n_init=0)
    with assert_raises(): _ = m.fit(x)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: BisectingKMeans lifecycle and native API")
