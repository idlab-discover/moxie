import numojo as nm
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.feature_selection import LinearRegressionImportance, SelectFromModel
from moxie.linear_model import LinearRegression


def main() raises:
    comptime dtype = DType.float64
    var X = nm.array[dtype]([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [4, 2])
    var y = nm.array[dtype]([4.0, 1.0, 4.0, 1.0], [4])
    var changed_y = nm.array[dtype]([1.0, 4.0, 1.0, 4.0], [4])
    var adapter = LinearRegressionImportance(LinearRegression[dtype](fit_intercept=False))
    with assert_raises(contains="NotFittedError"):
        _ = adapter.feature_importances()
    with assert_raises(contains="coef_"):
        _ = adapter.importances("feature_importances_")

    var weights = nm.array[dtype]([1.0, 2.0, 3.0, 4.0], [4])
    _ = adapter.fit(X, y, sample_weight=weights)
    assert_equal(adapter.n_features(), 2)
    var importances = adapter.importances("coef_")
    importances[nm.Item(0)] = 0
    assert_almost_equal(adapter.model.coef_[nm.Item(0)], 4.0)

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
    print("PASS LinearRegressionImportance fit, refit, clone, and fitted ownership")
