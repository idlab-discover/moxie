import numojo as nm
from std.math import inf, nan, isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import Lasso, ElasticNet
from moxie.feature_selection import SelectFromModel, LassoImportance, ElasticNetImportance, ModelThreshold, MaxFeatures, FeatureLimitPolicy
from moxie.feature_selection._model_importance import aggregate_importances
from moxie.feature_selection._from_model import importance_mask, importance_threshold

struct HalfFeatures(FeatureLimitPolicy):
    def __init__(out self):
        pass
    def resolve[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Int:
        return X.shape[1] // 2

def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(8, 4))
    var y = nm.zeros[dtype](nm.Shape(8))
    var beta: List[Float64] = [6, -3, 0.5, 0]
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, SIMD[dtype, 1](2 * beta[i % 4]))
    var model = Lasso[dtype](alpha=0.1, fit_intercept=False, tol=1e-7)
    var selector = SelectFromModel(LassoImportance(model.copy()))
    with assert_raises():
        _ = selector.get_support()
    selector.fit_named(X, y.copy(), [String("a"), String("b"), String("c"), String("d")])
    assert_false(model.__sklearn_is_fitted__())
    assert_false(selector.estimator.__sklearn_is_fitted__())
    assert_true(selector.__sklearn_is_fitted__())
    assert_equal(selector.threshold_(), 1e-5)
    assert_equal(selector.n_features_in_(), 4)
    assert_equal(selector.get_support(True)[List[Int]].__len__(), 3)
    var fitted = selector.estimator_()
    assert_almost_equal(Float64(fitted.model.coef_.unsafe_get(0)), 5.9, atol=1e-5)
    fitted.model.coef_.unsafe_set(0, 0)
    assert_equal(selector.get_support(True)[List[Int]].__len__(), 3)
    var fresh = clone(selector)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    _ = selector.set_params(threshold=ModelThreshold(String("1.1 * mean")))
    assert_equal(selector.get_support(True)[List[Int]].__len__(), 2)
    assert_equal(selector.get_feature_names_out()[1], "b")
    var result = selector.transform(X)
    var inverse = selector.inverse_transform(result)
    assert_equal(result.shape[1], 2)
    assert_equal(inverse.unsafe_get(10), 0)
    result.unsafe_set(0, 99)
    assert_equal(X.unsafe_get(0), 2)
    var saved = selector.estimator_().model.coef_.copy()
    var params = selector.get_params()
    params.max_features = MaxFeatures(1)
    params.threshold = ModelThreshold(-inf[DType.float64]())
    _ = selector.set_params(params)
    # Integer max_features was absent at fit, so its current value is used.
    assert_equal(selector.get_support(True)[List[Int]].__len__(), 1)
    assert_equal(selector.fit(X, y.copy()).max_features_(), 1)
    assert_equal(selector.estimator_().model.coef_.unsafe_get(0), saved.unsafe_get(0))
    var wrong = nm.zeros[dtype](nm.Shape(2))
    with assert_raises():
        _ = selector.fit(X, wrong^)
    assert_equal(selector.max_features_(), 1)
    var weights = nm.ones[dtype](nm.Shape(8))
    weights.unsafe_set(0, 3)
    var weighted = SelectFromModel(LassoImportance(model.copy()))
    _ = weighted.fit(X, y.copy(), sample_weight=weights)
    var direct = model.__sklearn_clone__()
    _ = direct.fit(X, y, sample_weight=weights)
    assert_equal(weighted.estimator_().model.coef_.unsafe_get(0), direct.coef_.unsafe_get(0))
    # Prefit construction permits transformation without fit, while learned
    # accessors appear only after fit. Fit must retain a deep fitted snapshot.
    var prefit = SelectFromModel(LassoImportance(direct.copy()), prefit=True)
    assert_equal(prefit.transform(X).shape[1], 3)
    with assert_raises():
        _ = prefit.threshold_()
    _ = prefit.fit(X)
    var coefficient = prefit.estimator_().model.coef_.unsafe_get(0)
    prefit.estimator.model.coef_.unsafe_set(0, 0)
    assert_equal(prefit.estimator_().model.coef_.unsafe_get(0), coefficient)
    assert_equal(prefit.transform(X).shape[1], 3)
    var cloned_prefit = clone(prefit)
    with assert_raises():
        _ = cloned_prefit.transform(X)
    var capped = SelectFromModel(LassoImportance(direct.copy()), prefit=True, max_features=HalfFeatures())
    with assert_raises():
        _ = capped.transform(X)
    _ = capped.fit(X)
    assert_equal(capped.max_features_(), 2)
    assert_equal(capped.transform(X).shape[1], 2)
    var empty = SelectFromModel(LassoImportance(direct.copy()), prefit=True, max_features=MaxFeatures(0))
    assert_equal(empty.transform(X).shape[1], 0)
    var too_many = SelectFromModel(LassoImportance(direct.copy()), prefit=True, max_features=MaxFeatures(100))
    _ = too_many.fit(X)
    assert_equal(too_many.max_features_(), 100)
    assert_equal(too_many.transform(X).shape[1], 3)
    var bad_limit = SelectFromModel(LassoImportance(direct.copy()), max_features=MaxFeatures(-1))
    with assert_raises():
        _ = bad_limit.fit(X, y.copy())
    var elastic = SelectFromModel(ElasticNetImportance(ElasticNet[dtype](alpha=0.1, fit_intercept=False)), threshold=ModelThreshold(String("median")), norm_order=2)
    var multi = nm.zeros[dtype](nm.Shape(8, 2))
    for i in range(8):
        multi.unsafe_set(i * 2, y.unsafe_get(i))
        multi.unsafe_set(i * 2 + 1, -2 * y.unsafe_get(i))
    _ = elastic.fit(X, multi^)
    assert_equal(elastic.estimator_().model.coef_.ndim, 2)
    assert_equal(elastic.transform(X).shape[1], 2)
    _ = elastic.set_params(threshold=ModelThreshold(String("bogus")))
    with assert_raises():
        _ = elastic.get_support()
    _ = elastic.set_params(threshold=None)
    assert_equal(elastic.transform(X).shape[1], 2)
    var bad_getter = SelectFromModel(LassoImportance(direct.copy()), prefit=True, importance_getter="missing.coef_")
    with assert_raises():
        _ = bad_getter.get_support()
    var bad_order = SelectFromModel(LassoImportance(direct.copy()), norm_order=0)
    with assert_raises():
        _ = bad_order.fit(X, y.copy())
    var invalid_threshold = SelectFromModel(LassoImportance(direct.copy()), threshold=ModelThreshold(nan[DType.float64]()))
    with assert_raises():
        _ = invalid_threshold.fit(X, y.copy())
    direct.warm_start = True
    direct.coef_.unsafe_set(0, 10000)
    var warm = SelectFromModel(LassoImportance(direct.copy()))
    _ = warm.fit(X, y.copy())
    assert_almost_equal(Float64(warm.estimator_().model.coef_.unsafe_get(0)), 5.9, atol=1e-5)
    var previous = warm.estimator_().model.coef_.unsafe_get(0)
    _ = warm.fit(X, nm.zeros[dtype](nm.Shape(8)))
    assert_equal(warm.estimator_().model.coef_.unsafe_get(0), 0)
    assert_true(previous > 0)
    var view = X.view_with_layout(nm.Shape(8, 4), nm.NDArrayStrides(-4, -1), 31)
    assert_equal(prefit.transform(view).shape[1], 3)
    X.unsafe_set(0, nan[dtype]())
    with assert_raises():
        _ = prefit.transform(X)

def kernels() raises:
    var values = nm.zeros[DType.float64](nm.Shape(2, 4))
    var data: List[Float64] = [3, 0, -4, 3, 4, 0, 3, 4]
    for i in range(8):
        values.unsafe_set(i, data[i])
    assert_equal(aggregate_importances(values, 2).unsafe_get(0), 5)
    assert_equal(aggregate_importances(values, inf[DType.float64]()).unsafe_get(2), 4)
    assert_equal(aggregate_importances(values, -inf[DType.float64]()).unsafe_get(2), 3)
    assert_equal(aggregate_importances(values, -1).unsafe_get(1), 0)
    assert_equal(aggregate_importances(values, square=True).unsafe_get(0), 25)
    var nonfinite = values.copy()
    nonfinite.unsafe_set(0, nan[DType.float64]())
    nonfinite.unsafe_set(4, inf[DType.float64]())
    assert_true(isnan(aggregate_importances(nonfinite, 3).unsafe_get(0)))
    assert_true(isnan(aggregate_importances(nonfinite, -3).unsafe_get(0)))
    var scores = aggregate_importances(values, 2)
    var mask = importance_mask(scores, 0, 2)
    assert_true(mask[0])
    assert_true(mask[2])
    assert_false(mask[3])
    assert_equal(importance_threshold(scores, ModelThreshold(String("median")), False), 5)
    scores.unsafe_set(0, nan[DType.float64]())
    mask = importance_mask(scores, 0, 2)
    assert_false(mask[0])
    assert_true(mask[2])
    assert_true(mask[3])
    assert_true(isnan(importance_threshold(scores, ModelThreshold(String("median")), False)))

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    kernels()
    print("SelectFromModel real-estimator contracts passed")
