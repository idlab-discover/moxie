"""Shared native-label CV contracts, instantiated separately to bound compilation."""
import numojo as nm
from moxie._labels import StringTarget
from moxie.feature_selection import DenseSupervisedCV, DecisionTreeClassifierImportance
from moxie.tree import DecisionTreeClassifier
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone


def main() raises:
    comptime E = DecisionTreeClassifierImportance[DType.float32, DType.int64, True]
    var adapter = DenseSupervisedCV[E, DType.float32]()
    var x = nm.zeros[DType.float32](nm.Shape(6, 2))
    var rows = nm.zeros[DType.int64](nm.Shape(3))
    rows[nm.Item(0)] = 5
    rows[nm.Item(1)] = 1
    rows[nm.Item(2)] = 1
    var y: E.Target
    y = StringTarget(["é", "a", "é", "a", "é", "z"])
    assert_equal(adapter.n_samples(x, y), 6)
    var sliced = adapter.target_rows(y, rows)
    assert_equal(sliced.get(0), "z")
    assert_equal(sliced.get(1), "a")
    sliced.set(1, "changed")
    assert_equal(y.get(1), "a")
    var matrix = StringTarget(["a", "x", "b", "y", "c", "z", "d", "w", "e", "u", "f", "v"], 6, 2)
    var taken = adapter.target_rows(matrix, rows)
    assert_equal(taken.rank(), 2)
    assert_equal(taken.rows(), 3)
    assert_equal(taken.columns(), 2)
    assert_equal(taken.get(0, 1), "v")
    assert_equal(taken.get(2, 0), "b")
    with assert_raises():
        _ = adapter.n_samples(x, StringTarget(["a"]))
    assert_equal(adapter.cv_labels(y).value().kind, "multiclass")
    var model = E(DecisionTreeClassifier[DType.int64, True](random_state=42))
    with assert_raises():
        _ = model.classes_()
    with assert_raises():
        _ = model.predict_proba(x)
    var probabilities = model.fit(x, y).predict_proba(x)
    assert_equal(probabilities.ndim, 2)
    assert_equal(probabilities.shape[0], 6)
    assert_equal(probabilities.shape[1], 3)
    var prediction = model.predict(x)
    assert_equal(prediction.rank(), 1)
    assert_equal(prediction.rows(), 6)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_true(model.__sklearn_is_fitted__())
    var params = model.get_params()
    params.max_depth = 1
    _ = fresh.set_params(params)
    assert_equal(fresh.get_params().max_depth.value(), 1)
    assert_false(Bool(model.get_params().max_depth))
    var owned = model.classes_()
    owned[0] = "changed"
    assert_equal(model.classes_()[0], "a")
    var snapshot = model.copy()
    snapshot.model.feature_importances_[nm.Item(0)] = 99
    assert_equal(model.feature_importances().unsafe_get(0), 0)
    snapshot.model.tree_.value.unsafe_set(0, 99)
    assert_true(model.model.tree_.value.unsafe_get(0) != 99)
    var weights = nm.ones[E.dtype](nm.Shape(6))
    adapter.fit(fresh, x, y, weights.copy())
    assert_true(fresh.__sklearn_is_fitted__())
    var weighted = adapter.metadata_rows(weights.copy(), rows)
    assert_equal(weighted.value().shape[0], 3)
    with assert_raises():
        _ = model.fit(x, matrix)
    assert_equal(model.n_features(), 2)
