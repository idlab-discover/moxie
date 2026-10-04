"""GraphicalLassoCV lifecycle, validation and native API adaptations."""
from numojo import Shape, Item, zeros, NDArrayStrides
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.covariance import GraphicalLassoCV, GraphicalLasso
from moxie.covariance._graph_lasso import AlphaGrid
from moxie.model_selection._indices import IndexSplit
from moxie.model_selection._split import KFold
from contracts import data


def check[dtype: DType]() raises:
    var X = data[dtype](53, 4)
    var model = GraphicalLassoCV[dtype]()
    var defaults = model.get_params()
    assert_equal(defaults.alphas.count, 4)
    assert_equal(defaults.n_refinements, 4)
    assert_false(Bool(defaults.cv))
    assert_false(Bool(defaults.n_jobs))
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.alpha_()
    with assert_raises():
        _ = model.cv_results_()
    model.fit(X)
    var result = model.cv_results_()
    assert_equal(result.alphas.size, 17)
    assert_equal(result.split_test_score.shape[1], 5)
    assert_equal(result.alphas.unsafe_get(16), 0)
    assert_true(model.alpha_() > 0)
    assert_equal(model.n_features_in_(), 4)
    var direct = GraphicalLasso[dtype](model.alpha_())
    direct.fit(X)
    assert_equal(model.error_norm(direct.covariance_()), 0)
    var saved_alpha = model.alpha_()
    result.alphas.unsafe_set(0, -17)
    assert_true(model.cv_results_().alphas.unsafe_get(0) >= 0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.alphas.count, 4)
    var snapshot = model.copy()
    var params = model.get_params()
    var explicit: List[Float64] = [0.03, 0.1, 0.03]
    params.alphas = AlphaGrid(explicit)
    params.cv = 3
    model.set_params(params)
    params.alphas.values.value()[0] = 77
    assert_equal(model.alphas.values.value()[0], 0.03)
    model.fit(X)
    assert_equal(model.cv_results_().alphas.size, 4)
    assert_equal(snapshot.alpha_(), saved_alpha)
    var cursor = KFold(3).split(X.shape[0])
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var custom = GraphicalLassoCV[dtype](alphas=explicit)
    custom.fit(X, splits=splits)
    assert_equal(custom.alpha_(), model.alpha_())
    assert_equal(custom.error_norm(model.covariance_()), 0)
    # Strided reads and independent fitted snapshots.
    var strided = X.view_with_layout(Shape(27, 4), NDArrayStrides(8, 1), 0)
    var smaller = GraphicalLassoCV[dtype](alphas=explicit, cv=3)
    smaller.fit(strided)
    assert_equal(smaller.n_features_in_(), 4)
    var prior = model.alpha_()
    model.alphas = AlphaGrid(0)
    with assert_raises():
        model.fit(X)
    assert_equal(model.alpha_(), prior)
    var invalid: List[Float64] = [0.0, 0.1]
    var bad = GraphicalLassoCV[dtype](alphas=invalid)
    with assert_raises():
        bad.fit(X)
    var lars = GraphicalLassoCV[dtype](mode="lars", alphas=explicit, cv=3)
    lars.fit(X)
    var fixed_lars = GraphicalLasso[dtype](lars.alpha_(), mode="lars")
    fixed_lars.fit(X)
    assert_equal(lars.error_norm(fixed_lars.covariance_()), 0)
    var lars_clone = clone(lars)
    assert_equal(lars_clone.mode, "lars")
    assert_false(lars_clone.__sklearn_is_fitted__())
    lars.fit(strided)
    assert_equal(lars.n_features_in_(), 4)
    bad = GraphicalLassoCV[dtype](n_jobs=2)
    with assert_raises():
        bad.fit(X)
    bad = GraphicalLassoCV[dtype](cv=54)
    with assert_raises():
        bad.fit(X)
    bad = GraphicalLassoCV[dtype]()
    with assert_raises():
        bad.fit(zeros[dtype](Shape(10, 1)))
    with assert_raises():
        bad.fit(zeros[dtype](Shape(10, 4)))
    with assert_raises():
        bad.fit(X, splits=List[IndexSplit]())
    splits[0].test[Item(0)] = 53
    with assert_raises():
        bad.fit(X, splits=splits)
    X[Item(0, 0)] = (Float64.MAX * 2).cast[dtype]()
    with assert_raises():
        bad.fit(X)


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
    print("PASS GraphicalLassoCV contracts (Float64/Float32)")
