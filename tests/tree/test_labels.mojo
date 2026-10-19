import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._labels import StringTarget
from moxie.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_text
from moxie.tree._classes import BaseDecisionTree
from moxie.tree._records import TreeTable, TreeColumn, Features

def strings[extra: Bool]() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 1))
    for i in range(6):
        X.unsafe_set(i, Float32(i))
    var y = StringTarget(["z", "", "é", "z", "", "é"])
    var model = BaseDecisionTree[True, extra, string_labels=True](random_state=2)
    with assert_raises():
        _ = model.predict(X)
    _ = model.fit(X, y)
    assert_equal(model.classes_[0][0], "")
    assert_equal(model.classes_[0][1], "z")
    assert_equal(model.classes_[0][2], "é")
    var prediction: StringTarget = model.predict(X)
    for i in range(6):
        assert_equal(prediction.get(i), y.get(i))
    assert_equal(model.score(X, y), 1.0)
    prediction.set(0, "independent")
    assert_equal(model.predict(X).get(0), "z")
    assert_equal(model.predict_proba(X)[0].shape[1], 3)
    assert_equal(model.predict_log_proba(X)[0].shape[0], 6)
    assert_true(export_text(model).byte_length() > 0)
    var path = model.cost_complexity_pruning_path(X, y)
    assert_true(path.ccp_alphas.size > 0)
    var clone = model.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    var weights = Dict[String, Float64]()
    weights["z"] = 2
    model.params.class_weight = weights^
    _ = model.fit(X, y)
    assert_equal(model.tree_.nodes[0].weighted_n_node_samples, 8.0)
    y.set(0, "changed")
    assert_equal(model.predict(X).get(0), "z")
    var matrix = StringTarget(["a", "x", "b", "y", "c", "x", "a", "y", "b", "x", "c", "y"], 6, 2)
    model.params.class_weight = NoneType()
    var first = Dict[String, Float64]()
    first["a"] = 2
    var second = Dict[String, Float64]()
    second["y"] = 3
    var class_weights: List[Dict[String, Float64]] = [first^, second^]
    model.params.class_weight = class_weights^
    _ = model.fit(X, matrix)
    assert_equal(model.tree_.nodes[0].weighted_n_node_samples, 16.0)
    assert_equal(model.predict(X).columns(), 2)
    assert_equal(model.predict(X).get(2, 0), "c")
    with assert_raises():
        _ = model.score(X, matrix)
    model.params.class_weight = NoneType()
    _ = model.fit(X, y)
    assert_equal(model.n_outputs_, 1)
    assert_equal(model.predict(X).get(0), "changed")

def numeric[extra: Bool, dtype: DType]() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 1))
    var y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(i, Float32(i))
        comptime if dtype == DType.int64:
            y.unsafe_set(i, Scalar[dtype](9007199254740992) + Scalar[dtype](i % 2))
        else:
            y.unsafe_set(i, Scalar[dtype](i % 2))
    var model = BaseDecisionTree[True, extra, dtype](random_state=2)
    _ = model.fit(X, y)
    var prediction: nm.NDArray[dtype] = model.predict(X)
    var classes: nm.NDArray[dtype] = model.classes_[0].copy()
    assert_equal(classes.size, 2)
    for i in range(6):
        assert_equal(prediction.unsafe_get(i), y.unsafe_get(i))
    assert_equal(model.score(X, y), 1.0)
    var strided = y.view_with_layout(nm.Shape(3), nm.NDArrayStrides(2), 0)
    var xv = X.view_with_layout(nm.Shape(3, 1), nm.NDArrayStrides(2, 1), 0)
    _ = model.fit(xv, strided)
    assert_equal(model.classes_[0].size, 1)
    assert_equal(model.predict(xv).unsafe_get(0), y.unsafe_get(0))
    comptime if dtype == DType.float32:
        y.unsafe_set(0, Scalar[dtype](0.5))
        with assert_raises():
            _ = model.fit(X, y)
        with assert_raises():
            _ = model.score(X, y)
        assert_equal(model.classes_[0].size, 1)

def main() raises:
    strings[False]()
    strings[True]()
    numeric[False, DType.int64]()
    numeric[True, DType.int64]()
    numeric[False, DType.float32]()
    numeric[True, DType.float32]()
    var named = DecisionTreeClassifier[string_labels=True]()
    var extra = ExtraTreeClassifier[label_dtype=DType.int64]()
    assert_true(not named.__sklearn_is_fitted__())
    assert_true(not extra.__sklearn_is_fitted__())
    print("PASS typed string/numeric labels, identity, probabilities, weights, refit and clone")
