import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.feature_selection import BayesianRidgeImportance, FeatureCount, RFE, SelectFromModel
from moxie.linear_model import BayesianRidge


def main() raises:
    comptime dtype = DType.float64
    var X = nm.array[dtype]([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], [4, 2])
    var y = nm.array[dtype]([4.0, 1.0, 7.9, 2.1], [4])
    var changed_y = nm.array[dtype]([1.0, 4.0, 2.1, 7.9], [4])
    var weights = nm.array[dtype]([1.0, 2.0, 3.0, 4.0], [4])
    var adapter = BayesianRidgeImportance(BayesianRidge[dtype](fit_intercept=False))
    with assert_raises(contains="NotFittedError"):
        _ = adapter.importances("auto")
    with assert_raises(contains="coef_"):
        _ = adapter.importances("feature_importances_")

    _ = adapter.fit(X, y, sample_weight=weights)
    assert_equal(adapter.n_features(), 2)
    var importances = adapter.importances("coef_")
    var coefficient = adapter.model.coef_[nm.Item(0)]
    importances[nm.Item(0)] = 0
    assert_equal(adapter.model.coef_[nm.Item(0)], coefficient)

    var fresh = clone(adapter)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_false(params.fit_intercept)
    params.fit_intercept = True
    _ = fresh.set_params(params)
    assert_true(fresh.get_params().fit_intercept)
    assert_false(adapter.get_params().fit_intercept)

    var prefit = SelectFromModel(adapter.copy(), prefit=True)
    assert_equal(prefit.get_support(True)[List[Int]], [0])
    _ = adapter.fit(X, changed_y)
    assert_equal(prefit.get_support(True)[List[Int]], [0])

    var selector = SelectFromModel(adapter^)
    with assert_raises(contains="requires y"):
        _ = selector.fit(X)
    _ = selector.fit(X, y.copy(), sample_weight=weights)
    assert_equal(selector.get_support(True)[List[Int]], [0])
    with assert_raises(contains="ValueError"):
        _ = selector.fit(X, nm.zeros[dtype](nm.Shape(3)))
    assert_equal(selector.get_support(True)[List[Int]], [0])
    _ = selector.fit(X, changed_y^)
    assert_equal(selector.get_support(True)[List[Int]], [1])
    assert_false(clone(selector).__sklearn_is_fitted__())

    var elimination = RFE(clone(fresh), n_features_to_select=FeatureCount(1))
    var column_target = y.reshape(nm.Shape(4, 1))
    _ = elimination.fit(X, column_target)
    var prediction = elimination.predict(X)
    var posterior = elimination.estimator_().model.predict_std(elimination.transform(X))
    assert_equal(prediction.ndim, 1)
    assert_equal(posterior[1].shape[0], X.shape[0])
    for i in range(X.shape[0]):
        assert_equal(posterior[0].unsafe_get(i), prediction.unsafe_get(i))
    print("PASS BayesianRidgeImportance fit, refit, clone, and fitted ownership")
