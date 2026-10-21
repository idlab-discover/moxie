import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import Ridge
from moxie.feature_selection import SelectFromModel, RFE, FeatureCount, RidgeImportance, ModelThreshold

# Orthogonal design: column j holds 2 on rows i % 4 == j, so a Ridge without
# intercept has the closed form coef_j = (X_j . y) / (X_j . X_j + alpha)
# = 8 beta_j / (8 + alpha), independently of the other columns.
def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(8, 4))
    var y = nm.zeros[dtype](nm.Shape(8))
    var beta: List[Float64] = [6, -3, 0.5, 0]
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, SIMD[dtype, 1](2 * beta[i % 4]))
    var tolerance = 1e-5 if dtype == DType.float32 else 1e-12
    var model = Ridge[dtype](alpha=1.0, fit_intercept=False)
    var adapter = RidgeImportance(model.copy())
    with assert_raises(contains="NotFittedError"):
        _ = adapter.feature_importances()
    assert_false(adapter.uses_l1_threshold())
    with assert_raises(contains="coef_"):
        _ = adapter.importances("feature_importances_")

    var selector = SelectFromModel(RidgeImportance(model.copy()))
    _ = selector.fit(X, y.copy())
    assert_false(selector.estimator.__sklearn_is_fitted__())
    var coef = selector.estimator_().model.coef_.copy()
    var mean = Float64(0)
    for j in range(4):
        assert_almost_equal(Float64(coef.unsafe_get(j)), 8 * beta[j] / 9, atol=tolerance)
        mean += abs(8 * beta[j] / 9) / 4
    # Ridge is not l1-penalized, so sklearn's default threshold is the mean.
    assert_almost_equal(selector.threshold_(), mean, atol=tolerance)
    assert_equal(selector.get_support(True)[List[Int]], [0, 1])
    assert_equal(selector.transform(X).shape[1], 2)
    _ = selector.set_params(threshold=ModelThreshold(String("0.1*mean")))
    assert_equal(selector.get_support(True)[List[Int]], [0, 1, 2])
    var fresh = clone(selector)
    assert_false(fresh.__sklearn_is_fitted__())

    var weights = nm.full[dtype](nm.Shape(8), SIMD[dtype, 1](2))
    var weighted = SelectFromModel(RidgeImportance(model.copy()))
    _ = weighted.fit(X, y.copy(), sample_weight=weights)
    assert_almost_equal(Float64(weighted.estimator_().model.coef_.unsafe_get(0)), 16 * 6 / 17.0, atol=tolerance * 10)

    var rfe = RFE(RidgeImportance(model.copy()), n_features_to_select=FeatureCount(1))
    _ = rfe.fit(X, y)
    assert_equal(rfe.ranking_(), [1, 2, 3, 4])
    assert_equal(rfe.support_(), [True, False, False, False])
    assert_almost_equal(Float64(rfe.predict(X).unsafe_get(0)), 2 * 8 * 6 / 9.0, atol=tolerance * 10)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS RidgeImportance adapter with SelectFromModel and RFE")
