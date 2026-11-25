"""Feature-name lifecycle for native kernel SVM table inputs."""

import numojo as nm
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.compose import ColumnTable
from moxie.svm import OneClassSVM, SVC


def main() raises:
    var names: List[String] = ["length", "width"]
    var reordered: List[String] = ["width", "length"]
    var labels: List[Int] = [1, 1, 2, 2]
    var storage = nm.zeros[DType.float32](nm.Shape(4, 3))
    var counts = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        storage.unsafe_set(i * 3 + 1, Float32(i) - 1.5)
        storage.unsafe_set(i * 3 + 2, Float32(i % 2))
        counts.unsafe_set(i, Int64(i % 2))
    var X = storage.view_with_layout(nm.Shape(4, 2), storage.strides, 1)
    var table = ColumnTable(X, names)
    var reversed_table = ColumnTable(X, reordered)
    var mixed_table = table.copy()
    mixed_table.add_strings("extra", ["a", "b", "c", "d"])

    var model = SVC[Int](kernel="linear", random_state=42)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(table)
    assert_true(model.fit(table, labels).predict(table) == labels)
    assert_true(model.feature_names_in_.value() == names)
    assert_true(not clone(model).feature_names_in_)
    with assert_raises(contains="same order"):
        _ = model.decision_function(reversed_table)
    with assert_raises(contains="requires numeric storage"):
        _ = model.fit(mixed_table, labels)
    assert_true(model.feature_names_in_.value() == names)
    assert_true(model.predict(table) == labels)
    assert_true(model.predict(X, feature_names=names) == labels)
    _ = model.fit(X, labels)
    assert_true(not model.feature_names_in_)

    var first = storage.view_with_layout(nm.Shape(4, 1), storage.strides, 1)
    var numeric_table = ColumnTable(first, ["length"])
    numeric_table.add_numeric("width", counts)
    var one_class = OneClassSVM(kernel="linear")
    var fit_labels = one_class.fit_predict(numeric_table)
    var predicted = one_class.predict(table)
    assert_equal(fit_labels.shape, predicted.shape)
    for i in range(fit_labels.size):
        assert_equal(fit_labels.item(i), predicted.item(i))
    assert_true(one_class.feature_names_in_.value() == names)
    print("PASS kernel SVM named-table lifecycle")
