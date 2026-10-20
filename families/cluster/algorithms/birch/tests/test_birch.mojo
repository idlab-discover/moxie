import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.cluster import Birch, IncrementalClusterer


def update[E: IncrementalClusterer](mut model: E, X: E.Input, y: Optional[E.Target]) raises:
    _ = model.partial_fit(X, y)


def check_tree[dtype: DType](model: Birch[dtype], index: Int, expected: Int) raises -> Int:
    var node = model.get_node(index)
    assert_true(len(node.subclusters_) <= node.branching_factor)
    assert_true(len(node.subclusters_) > 0)
    var total = 0
    var depth = 0
    for cluster in node.subclusters_:
        assert_true(cluster.n_samples_ > 0)
        total += cluster.n_samples_
        if node.is_leaf:
            assert_false(Bool(cluster.child_))
            assert_true(cluster.radius() <= node.threshold + 1e-4)
        else:
            assert_true(Bool(cluster.child_))
            var child_depth = check_tree(model, cluster.child_.value(), cluster.n_samples_)
            if depth == 0: depth = child_depth
            assert_equal(depth, child_depth)
    assert_equal(total, expected)
    return depth + 1


def check[dtype: DType]() raises:
    var model = Birch[dtype]()
    var params = model.get_params()
    assert_equal(params.threshold, 0.5)
    assert_equal(params.branching_factor, 50)
    assert_equal(params.n_clusters.value(), 3)
    assert_true(params.compute_labels)
    var x = nm.zeros[dtype](nm.Shape(36, 2))
    for i in range(36):
        x[nm.Item(i, 0)] = Scalar[dtype](i // 3)
        x[nm.Item(i, 1)] = Scalar[dtype](i % 3) * 0.01
    with assert_raises(): _ = model.predict(x)
    with assert_raises(): _ = model.transform(x)
    with assert_raises(): _ = model.partial_fit()
    with assert_raises(): _ = model.get_feature_names_out()
    _ = model.set_params(branching_factor=2, threshold=0.05).fit(x)
    assert_equal(model.subcluster_centers_.shape[0], 12)
    assert_equal(model.labels_.value().shape[0], 36)
    var predicted = model.predict(x)
    for i in range(36): assert_equal(predicted.item(i), model.labels_.value().item(i))
    var transformed = model.transform(x)
    assert_equal(transformed.shape[1], 12)
    assert_equal(transformed.shape[0], 36)
    _ = check_tree(model, model.root_.value(), 36)
    var seen = 0
    var previous = model.dummy_leaf_.value()
    var leaf = model.get_node(previous).next_leaf_
    while leaf:
        var node = model.get_node(leaf.value())
        assert_true(node.is_leaf)
        assert_equal(node.prev_leaf_.value(), previous)
        for cluster in node.subclusters_:
            for j in range(2): assert_equal(model.subcluster_centers_.item(seen, j), cluster.centroid_.item(j))
            seen += 1
        previous = leaf.value()
        leaf = node.next_leaf_
    assert_equal(seen, 12)
    var names = model.get_feature_names_out()
    assert_equal(names[0], "birch0")
    assert_equal(names[11], "birch11")
    var wrong_names: List[String] = ["one"]
    with assert_raises(): _ = model.get_feature_names_out(wrong_names.copy())
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(Bool(fresh.labels_))
    assert_equal(fresh.params.branching_factor, 2)
    _ = fresh.set_params(params)
    params.threshold = 42
    assert_equal(fresh.params.threshold, 0.5)
    var old_center = model.subcluster_centers_.item(0, 0)
    var copied = model.copy()
    copied.subcluster_centers_[nm.Item(0, 0)] = 1234
    assert_equal(model.subcluster_centers_.item(0, 0), old_center)
    _ = copied.partial_fit(x)
    _ = check_tree(model, model.root_.value(), 36)
    _ = check_tree(copied, copied.root_.value(), 72)
    var node_copy = model.get_node(model.root_.value())
    node_copy.subclusters_[0].centroid_.unsafe_set(0, 999)
    assert_true(model.get_node(model.root_.value()).subclusters_[0].centroid_.item(0) != 999)

    var incremental = Birch[dtype](threshold=0.05, branching_factor=2, n_clusters=None)
    for start in range(0, 36, 6):
        var batch = nm.zeros[dtype](nm.Shape(6, 2))
        for i in range(6):
            for j in range(2): batch[nm.Item(i, j)] = x.item(start+i, j)
        update(incremental, batch, Optional[nm.NDArray[dtype]](None))
        assert_equal(incremental.labels_.value().size, 6)
    assert_equal(incremental.subcluster_centers_.shape[0], 12)
    for i in range(12):
        for j in range(2): assert_equal(incremental.subcluster_centers_.item(i,j), model.subcluster_centers_.item(i,j))
    var last_labels = incremental.labels_.value().copy()
    _ = incremental.set_params(n_clusters=Optional[Int](3)).partial_fit()
    for i in range(12): assert_equal(incremental.subcluster_labels_.item(i), model.subcluster_labels_.item(i))
    for i in range(6): assert_equal(incremental.labels_.value().item(i), last_labels.item(i))
    var wrong = nm.zeros[dtype](nm.Shape(3, 1))
    with assert_raises(): _ = incremental.partial_fit(wrong)
    with assert_raises(): _ = model.predict(wrong)
    with assert_raises(): _ = model.transform(wrong)
    _ = check_tree(incremental, incremental.root_.value(), 36)

    var unlabeled = Birch[dtype](compute_labels=False, n_clusters=None)
    _ = unlabeled.partial_fit(x)
    assert_false(Bool(unlabeled.labels_))
    _ = unlabeled.predict(x)
    _ = unlabeled.set_params(compute_labels=True).partial_fit(x)
    var saved = unlabeled.labels_.value().copy()
    _ = unlabeled.set_params(compute_labels=False).fit(wrong)
    for i in range(saved.size): assert_equal(unlabeled.labels_.value().item(i), saved.item(i))
    assert_equal(unlabeled.n_features_in_, 1)
    var singleton = Birch[dtype]()
    _ = singleton.fit(wrong)
    assert_true(singleton.n_clusters_warning_)
    assert_equal(singleton.subcluster_centers_.shape[0], 1)
    assert_equal(singleton.subcluster_labels_.item(0), 0)
    _ = singleton.set_params(n_clusters=Optional[Int](1))
    with assert_raises(): _ = singleton.partial_fit()
    _ = model.set_params(threshold=0.0)
    with assert_raises(): _ = model.fit(x)
    _ = check_tree(model, model.root_.value(), 36)
    _ = model.set_params(threshold=0.5, branching_factor=1)
    with assert_raises(): _ = model.fit(x)
    _ = model.set_params(branching_factor=2, n_clusters=Optional[Int](0))
    with assert_raises(): _ = model.partial_fit()
    _ = model.set_params(n_clusters=Optional[Int](None))
    var empty = nm.zeros[dtype](nm.Shape(0,2))
    with assert_raises(): _ = model.fit(empty)
    var nonfinite = nm.zeros[dtype](nm.Shape(2,2))
    nonfinite[nm.Item(0,0)] = Scalar[dtype](Float64("nan"))
    with assert_raises(): _ = model.partial_fit(nonfinite)
    _ = model.fit_predict(x)
    _ = model.fit_transform(x)
    var input_names: List[String] = ["a", "b"]
    _ = model.get_feature_names_out(input_names.copy())
    var boundary = nm.zeros[dtype](nm.Shape(2,1))
    boundary[nm.Item(1,0)] = 1
    var inclusive = Birch[dtype](threshold=0.5, n_clusters=None)
    _ = inclusive.fit(boundary)
    assert_equal(inclusive.subcluster_centers_.shape[0], 1)
    assert_equal(inclusive.subcluster_centers_.item(0,0), 0.5)
    _ = inclusive.set_params(threshold=0.499).fit(boundary)
    assert_equal(inclusive.subcluster_centers_.shape[0], 2)
    boundary[nm.Item(0,0)] = 99
    assert_equal(inclusive.subcluster_centers_.item(0,0), 0)
    var overflow = nm.zeros[dtype](nm.Shape(2,1))
    comptime if dtype == DType.float32: overflow[nm.Item(0,0)] = Scalar[dtype](1e30)
    else: overflow[nm.Item(0,0)] = Scalar[dtype](1e200)
    with assert_raises(): _ = inclusive.partial_fit(overflow)
    assert_equal(inclusive.subcluster_centers_.shape[0], 2)
    _ = check_tree(inclusive, inclusive.root_.value(), 2)
    print("PASS: Birch lifecycle, tree invariants and native API", String(dtype))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
