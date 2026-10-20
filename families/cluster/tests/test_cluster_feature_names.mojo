"""Centroid transform names follow fitted state, including incremental fits."""
import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.base import clone
from moxie.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

def check_names(names: List[String], prefix: String, count: Int) raises:
    assert_equal(len(names), count)
    for i in range(count):
        assert_equal(names[i], prefix + String(i))
        print(prefix, count, names[i])

def check_kmeans[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    for i in range(6): x[nm.Item(i, 0)] = Scalar[dtype](i)
    var model = KMeans[dtype](2, random_state=0)
    with assert_raises(): _ = model.get_feature_names_out()
    _ = model.fit(x)
    check_names(model.get_feature_names_out(), "kmeans", 2)
    var inputs: List[String] = ["left", "right"]
    check_names(model.get_feature_names_out(inputs.copy()), "kmeans", 2)
    var bad: List[String] = ["left"]
    with assert_raises(): _ = model.get_feature_names_out(bad.copy())
    with assert_raises(): _ = model.get_feature_names_out(List[String]())
    var names = model.get_feature_names_out()
    names[0] = "changed"
    assert_equal(model.get_feature_names_out()[0], "kmeans0")
    var fresh = clone(model)
    with assert_raises(): _ = fresh.get_feature_names_out()
    _ = model.set_params(n_clusters=3)
    check_names(model.get_feature_names_out(), "kmeans", 2)
    assert_equal(model.transform(x).shape[1], 2)
    var wider = nm.zeros[dtype](nm.Shape(6, 3))
    for i in range(6): wider[nm.Item(i, 0)] = Scalar[dtype](i)
    _ = model.fit(wider)
    check_names(model.get_feature_names_out(), "kmeans", 3)
    assert_equal(model.transform(wider).shape[1], 3)
    with assert_raises(): _ = model.get_feature_names_out(inputs.copy())

def check_minibatchkmeans[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    for i in range(6): x[nm.Item(i, 0)] = Scalar[dtype](i)
    var model = MiniBatchKMeans[dtype](2, random_state=0)
    with assert_raises(): _ = model.get_feature_names_out()
    _ = model.fit(x)
    check_names(model.get_feature_names_out(), "minibatchkmeans", 2)
    var inputs: List[String] = ["left", "right"]
    check_names(model.get_feature_names_out(inputs.copy()), "minibatchkmeans", 2)
    var bad: List[String] = ["left"]
    with assert_raises(): _ = model.get_feature_names_out(bad.copy())
    with assert_raises(): _ = model.get_feature_names_out(List[String]())
    var names = model.get_feature_names_out()
    names[0] = "changed"
    assert_equal(model.get_feature_names_out()[0], "minibatchkmeans0")
    var fresh = clone(model)
    with assert_raises(): _ = fresh.get_feature_names_out()
    _ = model.set_params(n_clusters=3)
    check_names(model.get_feature_names_out(), "minibatchkmeans", 2)
    assert_equal(model.transform(x).shape[1], 2)
    var wider = nm.zeros[dtype](nm.Shape(6, 3))
    for i in range(6): wider[nm.Item(i, 0)] = Scalar[dtype](i)
    _ = model.fit(wider)
    check_names(model.get_feature_names_out(), "minibatchkmeans", 3)
    assert_equal(model.transform(wider).shape[1], 3)
    with assert_raises(): _ = model.get_feature_names_out(inputs.copy())
    var incremental = MiniBatchKMeans[dtype](2, random_state=0, compute_labels=False)
    _ = incremental.partial_fit(x)
    check_names(incremental.get_feature_names_out(), "minibatchkmeans", 2)
    _ = incremental.partial_fit(x)
    check_names(incremental.get_feature_names_out(inputs.copy()), "minibatchkmeans", 2)

def check_bisectingkmeans[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    for i in range(6): x[nm.Item(i, 0)] = Scalar[dtype](i)
    var model = BisectingKMeans[dtype](2, random_state=0)
    with assert_raises(): _ = model.get_feature_names_out()
    _ = model.fit(x)
    check_names(model.get_feature_names_out(), "bisectingkmeans", 2)
    var inputs: List[String] = ["left", "right"]
    check_names(model.get_feature_names_out(inputs.copy()), "bisectingkmeans", 2)
    var bad: List[String] = ["left"]
    with assert_raises(): _ = model.get_feature_names_out(bad.copy())
    with assert_raises(): _ = model.get_feature_names_out(List[String]())
    var names = model.get_feature_names_out()
    names[0] = "changed"
    assert_equal(model.get_feature_names_out()[0], "bisectingkmeans0")
    var fresh = clone(model)
    with assert_raises(): _ = fresh.get_feature_names_out()
    _ = model.set_params(n_clusters=3)
    check_names(model.get_feature_names_out(), "bisectingkmeans", 2)
    assert_equal(model.transform(x).shape[1], 2)
    var wider = nm.zeros[dtype](nm.Shape(6, 3))
    for i in range(6): wider[nm.Item(i, 0)] = Scalar[dtype](i)
    _ = model.fit(wider)
    check_names(model.get_feature_names_out(), "bisectingkmeans", 3)
    assert_equal(model.transform(wider).shape[1], 3)
    with assert_raises(): _ = model.get_feature_names_out(inputs.copy())

def main() raises:
    check_kmeans[DType.float32]()
    check_kmeans[DType.float64]()
    check_minibatchkmeans[DType.float32]()
    check_minibatchkmeans[DType.float64]()
    check_bisectingkmeans[DType.float32]()
    check_bisectingkmeans[DType.float64]()
