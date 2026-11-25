import numojo as nm
from moxie.base import clone
from moxie.feature_selection import LogisticRegressionImportance, SelectFromModel
from moxie.linear_model import LogisticRegression
from std.testing import assert_equal, assert_false, assert_raises


def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 3))
    var y = nm.zeros[DType.int64](nm.Shape(6))
    for i in range(6):
        X[nm.Item(i, 0)] = Float32(i)
        X[nm.Item(i, 1)] = Float32((i + 1) % 2)
        y[nm.Item(i)] = Int64(2**53 + i // 3)
    var estimator = LogisticRegressionImportance(LogisticRegression[DType.float32](C=2.0))
    comptime E = LogisticRegressionImportance[DType.float32]
    comptime assert type_of(estimator) == E

    with assert_raises(contains="NotFittedError"):
        _ = estimator.feature_importances()
    with assert_raises(contains="NotFittedError"):
        _ = estimator.classes_()
    with assert_raises(contains="requires y"):
        estimator.fit_selection(X, None, None)
    var bad_weights = nm.zeros[DType.float32](nm.Shape(5))
    with assert_raises(contains="ValueError"):
        _ = estimator.fit(X, y, sample_weight=bad_weights)
    assert_false(estimator.__sklearn_is_fitted__())

    var weights = nm.full[DType.float32](nm.Shape(6), 1)
    var prediction = estimator.fit(X, y, sample_weight=weights).predict(X)
    assert_equal(prediction[nm.Item(0)], Int64(2**53))
    assert_equal(prediction[nm.Item(5)], Int64(2**53 + 1))
    var classes = estimator.classes_()
    classes[nm.Item(0)] = 3
    assert_equal(estimator.classes_()[nm.Item(0)], Int64(2**53))
    var importances = estimator.importances("coef_")
    var original = importances[nm.Item(0, 0)]
    importances[nm.Item(0, 0)] = 0
    assert_equal(estimator.importances("auto")[nm.Item(0, 0)], original)
    with assert_raises(contains="ValueError"):
        _ = estimator.importances("feature_importances_")

    var fresh = clone(estimator)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_equal(params.C, 2.0)
    params.C = 0.5
    assert_equal(fresh.set_params(params).get_params().C, 0.5)

    var selector = SelectFromModel(estimator^, prefit=True)
    _ = selector.fit(X)
    selector.estimator.model.coef_[nm.Item(0, 0)] = 0
    assert_equal(selector.estimator_().feature_importances()[nm.Item(0, 0)], original)
    for i in range(6):
        y[nm.Item(i)] = Int64(i // 3)
    _ = selector.estimator.fit(X, y)
    assert_equal(selector.estimator.classes_()[nm.Item(0)], Int64(0))
    assert_equal(selector.estimator_().classes_()[nm.Item(0)], Int64(2**53))
    print("PASS LogisticRegressionImportance contracts")
