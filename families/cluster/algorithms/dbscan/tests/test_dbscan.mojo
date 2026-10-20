import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.cluster import DBSCAN, dbscan, Clusterer
from moxie.neighbors._distance import Metric


def dispatch[T: Clusterer](mut model: T, X: T.Input, y: Optional[T.Target]) raises -> T.Labels:
    return model.fit_predict(X, y)


def check[dtype: DType]() raises:
    var defaults = DBSCAN[dtype]()
    assert_equal(defaults.params.eps, Float64(0.5))
    assert_equal(defaults.params.min_samples, 5)
    assert_equal(defaults.params.metric, "euclidean")
    assert_equal(defaults.params.algorithm, "auto")
    assert_equal(defaults.params.leaf_size, 30)
    assert_false(Bool(defaults.params.p))
    assert_false(Bool(defaults.params.n_jobs))
    assert_false(Bool(defaults.params.metric_params))
    var x = nm.zeros[dtype](nm.Shape(7, 1))
    var values: List[Int] = [0, 2, 3, 4, 6, 8, 10]
    for i in range(7): x[nm.Item(i,0)] = SIMD[dtype,1](values[i])
    var model = DBSCAN[dtype](1, min_samples=3)
    assert_false(model.__sklearn_is_fitted__())
    var labels = dispatch(model, x, Optional[nm.NDArray[dtype]](None))
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.core_sample_indices_.size, 1)
    assert_equal(model.core_sample_indices_.item(0), Int64(2))
    assert_equal(model.components_.item(0,0), SIMD[dtype,1](3))
    for i in range(7): assert_equal(labels.item(i), Int64(0 if i >= 1 and i <= 3 else -1))
    labels[nm.Item(2)] = 99
    assert_equal(model.labels_.item(2), Int64(0))
    x[nm.Item(2,0)] = 100
    assert_equal(model.components_.item(0,0), SIMD[dtype,1](3))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.labels_.size, 0)
    assert_equal(fresh.get_params().eps, Float64(1))
    _ = model.set_params(min_samples=10)
    assert_true(model.__sklearn_is_fitted__())
    _ = model.fit(x)
    assert_equal(model.core_sample_indices_.size, 0)
    assert_equal(model.components_.shape[0], 0)
    assert_equal(model.components_.shape[1], 1)
    for i in range(7): assert_equal(model.labels_.item(i), Int64(-1))
    var params = model.get_params()
    params.min_samples = 1
    _ = model.set_params(params)
    assert_equal(model.get_params().min_samples, 1)
    var helper = dbscan(x, min_samples=1)
    assert_equal(helper.core_sample_indices.size, 7)
    var tiny = nm.zeros[dtype](nm.Shape(2,2))
    var weights = nm.zeros[dtype](nm.Shape(2))
    weights[nm.Item(0)] = 6
    weights[nm.Item(1)] = -1
    _ = model.set_params(min_samples=6)
    _ = model.fit(tiny, sample_weight=weights.copy())
    assert_equal(model.core_sample_indices_.size, 0)
    weights[nm.Item(1)] = 0
    var weighted = model.fit_predict(tiny, sample_weight=weights.copy())
    assert_equal(model.core_sample_indices_.size, 2)
    assert_equal(weighted.item(1), Int64(0))
    assert_equal(model.n_features_in_, 2)
    # Both negative strides and weights are logical views, not flat buffers.
    var reverse = x.view_with_layout(nm.Shape(7,1), nm.Strides(-1,1), 6)
    var reversed_labels = dbscan(reverse, min_samples=1)
    assert_equal(reversed_labels.labels.item(0), Int64(0))
    var row_weights = nm.zeros[dtype](nm.Shape(7))
    row_weights[nm.Item(0)] = 6
    var reverse_weights = row_weights.view_with_layout(nm.Shape(7), nm.Strides(-1), 6)
    _ = model.fit(reverse, sample_weight=reverse_weights^)
    assert_equal(model.core_sample_indices_.size, 1)
    assert_equal(model.core_sample_indices_.item(0), Int64(6))
    assert_equal(model.components_.item(0,0), SIMD[dtype,1](0))
    var bad = nm.zeros[dtype](nm.Shape(1))
    with assert_raises(): _ = model.fit(bad)
    assert_equal(model.core_sample_indices_.item(0), Int64(6))
    with assert_raises(): _ = model.fit(tiny, sample_weight=bad.copy())
    var empty = nm.zeros[dtype](nm.Shape(0,2))
    with assert_raises(): _ = model.fit(empty)
    weights[nm.Item(0)] = SIMD[dtype,1](Float64("nan"))
    with assert_raises(): _ = model.fit(tiny, sample_weight=weights.copy())
    tiny[nm.Item(0,0)] = SIMD[dtype,1](Float64("inf"))
    with assert_raises(): _ = model.fit(tiny)
    tiny[nm.Item(0,0)] = 0
    _ = model.set_params(eps=Float64(0))
    with assert_raises(): _ = model.fit(tiny)
    _ = model.set_params(eps=Float64(1), min_samples=0)
    with assert_raises(): _ = model.fit(tiny)
    _ = model.set_params(min_samples=1, algorithm="unknown")
    with assert_raises(): _ = model.fit(tiny)
    _ = model.set_params(algorithm="brute", leaf_size=0)
    with assert_raises(): _ = model.fit(tiny)
    _ = model.set_params(leaf_size=30, metric="precomputed")
    with assert_raises(): _ = model.fit(x)
    tiny[nm.Item(0,0)] = -1
    with assert_raises(): _ = model.fit(tiny)
    tiny[nm.Item(0,0)] = 0
    _ = model.set_params(algorithm="kd_tree")
    with assert_raises(): _ = model.fit(tiny)
    var metric_weights = nm.zeros[dtype](nm.Shape(2))
    metric_weights[nm.Item(0)] = 1
    var metric = Metric[dtype]("minkowski", p=3, w=metric_weights.copy())
    var feature_data = nm.zeros[dtype](nm.Shape(3,2))
    feature_data[nm.Item(1,1)] = 100
    feature_data[nm.Item(2,0)] = 2
    var feature_result = dbscan(feature_data, 1, min_samples=2, metric="minkowski", metric_params=metric.copy())
    assert_equal(feature_result.labels.item(0), Int64(0))
    assert_equal(feature_result.labels.item(1), Int64(0))
    assert_equal(feature_result.labels.item(2), Int64(-1))
    var configured = DBSCAN[dtype](metric="minkowski", metric_params=metric.copy(), min_samples=1)
    _ = configured.fit(tiny)
    assert_equal(configured.core_sample_indices_.size, 2)
    var configured_clone = clone(configured)
    assert_false(configured_clone.__sklearn_is_fitted__())
    assert_equal(configured_clone.get_params().metric_params.value().p, Float64(3))
    _ = configured.set_params(metric="cosine")
    var clear_metric = Optional[Optional[Metric[dtype]]](Optional[Metric[dtype]](None))
    _ = configured.set_params(metric_params=clear_metric)
    with assert_raises(): _ = configured.fit(tiny)
    _ = configured.set_params(metric="euclidean", n_jobs=Optional[Optional[Int]](Optional[Int](0)))
    with assert_raises(): _ = configured.fit(tiny)
    _ = configured.set_params(n_jobs=Optional[Optional[Int]](Optional[Int](None)))
    _ = configured.fit(tiny, y=nm.zeros[dtype](nm.Shape(19)))
    assert_true(configured.__sklearn_is_fitted__())
    print("PASS DBSCAN contracts", dtype)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
