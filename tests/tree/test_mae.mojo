import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree._classes import BaseDecisionTree
from moxie.tree._criterion import MAE
from moxie.tree._records import CriterionConfig, TrainingData

def lifecycle[extra: Bool]() raises:
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i))
    var model = BaseDecisionTree[False, extra](criterion="absolute_error", random_state=3)
    with assert_raises():
        _ = model.predict(X)
    _ = model.fit(X, y)
    for i in range(4):
        assert_equal(model.predict(X).unsafe_get(i), y.unsafe_get(i))
    var clone = model.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    assert_equal(clone.params.criterion, "absolute_error")
    y.unsafe_set(0, 7)
    _ = model.fit(X, y)
    assert_equal(model.predict(X).unsafe_get(0), 7.0)
    X.unsafe_set(0, Float32(Float64("nan")))
    with assert_raises():
        _ = model.fit(X, y)
    with assert_raises():
        _ = model.predict(X)
    X.unsafe_set(0, 0)
    assert_equal(model.predict(X).unsafe_get(0), 7.0)
    model.params.criterion = "squared_error"
    _ = model.fit(X, y)
    assert_equal(model.predict(X).unsafe_get(0), 7.0)

def main() raises:
    lifecycle[False]()
    lifecycle[True]()
    var y = nm.zeros[DType.float64](nm.Shape(4, 1))
    var samples = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        y.unsafe_set(i, Float64(i))
        samples.unsafe_set(i, Int64(i))
    var data = TrainingData(nm.zeros[DType.float32](nm.Shape(4, 1)), y^, None,
        nm.zeros[DType.uint8](nm.Shape(1)), nm.full[DType.int64](nm.Shape(1), -1),
        nm.zeros[DType.int8](nm.Shape(1)))
    var criterion = MAE(CriterionConfig(1, [1], 4))
    criterion.initialize(data, samples, 0, 4, 4)
    assert_equal(criterion.node_impurity(), 1.0)
    criterion.initialize(data, samples, 1, 3, 4)
    assert_equal(criterion.node_impurity(), 0.5)
    print("PASS MAE offset correction and both regressor lifecycles")
