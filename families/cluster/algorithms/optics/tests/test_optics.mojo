import numojo as nm
from std.math import nan, inf
from moxie.neighbors._distance import Metric
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie import clone
from moxie.cluster import Clusterer, OPTICS, compute_optics_graph, cluster_optics_dbscan, cluster_optics_xi
from moxie.cluster._optics import OPTICSSize


def check[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(8, 1))
    var values = [1, 2, 3, 1, 8, 8, 7, 100]
    for i in range(8): x[nm.Item(i, 0)] = SIMD[dtype, 1](values[i])
    var model = OPTICS[dtype](min_samples=OPTICSSize(3))
    assert_false(model.__sklearn_is_fitted__())
    _ = generic_fit(model, x, Optional[nm.NDArray[dtype]](None))
    var labels = model.fit_predict(x)
    assert_true(model.__sklearn_is_fitted__())
    var expected = [0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(8): assert_equal(labels.item(i), Int64(expected[i]))
    assert_true(Bool(model.cluster_hierarchy_))
    var graph = compute_optics_graph(x, min_samples=OPTICSSize(3), max_eps=inf[DType.float64](),
        metric="minkowski", p=2, metric_params=None, algorithm="auto", leaf_size=30, n_jobs=None)
    for i in range(8):
        assert_equal(graph.ordering.item(i), model.ordering_.item(i))
        assert_equal(graph.core_distances.item(i), model.core_distances_.item(i))
        assert_equal(graph.reachability.item(i), model.reachability_.item(i))
        assert_equal(graph.predecessor.item(i), model.predecessor_.item(i))
    var copied = clone(model)
    assert_false(copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().min_samples.value, Float64(3))
    labels.unsafe_set(0, -1)
    assert_equal(model.labels_.item(0), Int64(0))
    _ = model.set_params(cluster_method=String("dbscan"), eps=Optional[Float64](2)).fit(x)
    assert_false(Bool(model.cluster_hierarchy_))
    var direct = cluster_optics_dbscan(reachability=model.reachability_, core_distances=model.core_distances_, ordering=model.ordering_, eps=2)
    for i in range(8): assert_equal(direct.item(i), model.labels_.item(i))
    var params = model.get_params()
    params.min_samples = OPTICSSize(Float64(0.375))
    _ = model.set_params(params).fit(x)
    for i in range(8): assert_equal(model.labels_.item(i), direct.item(i))
    with assert_raises():
        _ = model.set_params(min_samples=OPTICSSize(1)).fit(x)
    with assert_raises():
        _ = model.set_params(min_samples=OPTICSSize(9)).fit(x)
    with assert_raises():
        _ = model.set_params(min_samples=OPTICSSize(3), max_eps=Float64(1), eps=Optional[Float64](2)).fit(x)
    with assert_raises():
        var bad = OPTICS[dtype](metric="cosine")
        _ = bad.fit(x)
    with assert_raises():
        var bad = OPTICS[dtype](memory=String("cache"))
        _ = bad.fit(x)
    var order = model.ordering_.copy()
    order.unsafe_set(0, order.item(1))
    with assert_raises():
        _ = cluster_optics_dbscan(reachability=model.reachability_, core_distances=model.core_distances_, ordering=order, eps=2)
    var zeros = nm.zeros[dtype](nm.Shape(5, 2))
    var duplicate = OPTICS[dtype](max_eps=0)
    _ = duplicate.fit(zeros)
    for i in range(5):
        assert_equal(duplicate.labels_.item(i), Int64(0))
        assert_equal(duplicate.ordering_.item(i), Int64(i))
    assert_equal(duplicate.cluster_hierarchy_.value().shape[1], 2)


def main() raises:
    test_helpers()
    test_layout_and_metric()
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: OPTICS lifecycle, extraction and validation")


def test_helpers() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 1))
    for i in range(6): x[nm.Item(i, 0)] = Float64(i)
    var graph = compute_optics_graph(x, min_samples=OPTICSSize(2), max_eps=Float64(2),
        metric="euclidean", p=2, metric_params=None, algorithm="brute", leaf_size=30, n_jobs=None)
    for i in range(6):
        assert_equal(graph.ordering.item(i), Int64(i))
        assert_equal(graph.core_distances.item(i), Float64(1))
        assert_equal(graph.predecessor.item(i), Int64(i - 1))
    var xi = cluster_optics_xi(reachability=graph.reachability, predecessor=graph.predecessor,
        ordering=graph.ordering, min_samples=OPTICSSize(2))
    for i in range(6): assert_equal(xi.labels.item(i), Int64(0))
    assert_equal(xi.clusters.item(0, 0), Int64(0))
    assert_equal(xi.clusters.item(0, 1), Int64(5))
    with assert_raises():
        _ = cluster_optics_xi(reachability=graph.reachability, predecessor=graph.predecessor,
            ordering=graph.ordering, min_samples=OPTICSSize(2), xi=1)
    var model = OPTICS(min_samples=3)
    _ = model.fit(x)
    var before = model.labels_.copy()
    var bad = nm.zeros[DType.float64](nm.Shape(0, 2))
    with assert_raises(): _ = model.fit(bad)
    assert_equal(model.n_features_in_, 1)
    for i in range(6): assert_equal(model.labels_.item(i), before.item(i))
    var smaller = nm.zeros[DType.float64](nm.Shape(4, 3))
    _ = model.fit(smaller)
    assert_equal(model.labels_.shape[0], 4)
    assert_equal(model.n_features_in_, 3)


def test_layout_and_metric() raises:
    var storage = nm.zeros[DType.float64](nm.Shape(12, 4))
    for i in range(12):
        storage[nm.Item(i, 0)] = Float64(i)
        storage[nm.Item(i, 2)] = Float64(i % 3)
    var x = storage.view_with_layout(nm.Shape(6, 2), nm.Strides(8, 2), 0)
    var dense = nm.zeros[DType.float64](nm.Shape(6, 2))
    var scaled = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        for j in range(2):
            dense[nm.Item(i, j)] = x.item(i, j)
            scaled[nm.Item(i, j)] = x.item(i, j) * Float64(2 if j == 0 else 1)
    var weights = nm.ones[DType.float64](nm.Shape(2))
    weights.unsafe_set(0, 4)
    var metric = Metric[DType.float64]("minkowski", 2, w=weights.copy())
    var model = OPTICS(min_samples=3, p=3, metric_params=metric.copy())
    _ = model.fit(x)
    var comparison = OPTICS(min_samples=3, metric="euclidean")
    _ = comparison.fit(scaled)
    for i in range(6):
        assert_equal(model.ordering_.item(i), comparison.ordering_.item(i))
        assert_equal(model.core_distances_.item(i), comparison.core_distances_.item(i))
        for j in range(2): assert_equal(x.item(i, j), dense.item(i, j))
    var copied = clone(model)
    var parameters = copied.get_params()
    parameters.metric_params.value().w.value().unsafe_set(0, 1)
    assert_equal(model.params.metric_params.value().w.value().item(0), Float64(4))
    var empty = nm.zeros[DType.float64](nm.Shape(0, 2))
    var vector = nm.zeros[DType.float64](nm.Shape(3))
    var invalid = dense.copy()
    invalid[nm.Item(0, 0)] = nan[DType.float64]()
    with assert_raises(): _ = comparison.fit(empty)
    with assert_raises(): _ = comparison.fit(vector)
    with assert_raises(): _ = comparison.fit(invalid)
    with assert_raises(): _ = comparison.set_params(n_jobs=Optional[Int](0)).fit(dense)
    with assert_raises(): _ = comparison.set_params(n_jobs=Optional[Int](1), leaf_size=0).fit(dense)
    with assert_raises(): _ = comparison.set_params(leaf_size=30, metric=String("precomputed")).fit(dense)
    with assert_raises(): _ = comparison.set_params(metric=String("euclidean"), min_cluster_size=Optional[OPTICSSize](OPTICSSize(Float64(0)))).fit(dense)


def generic_fit[E: Clusterer & Deinitable](mut model: E, x: E.Input, y: Optional[E.Target]) raises -> Bool where conforms_to(E.Params, Deinitable):
    _ = model.set_params(model.get_params()).fit(x, y)
    return model.__sklearn_is_fitted__()
