"""Fluent borrows, default metadata, explicit weights, and clone ownership."""
from search_support import Adapter, Scorer, DummyRegressor, Dense, data, grid
from moxie.base import SupervisedScorer, RegressorMixin, clone
from moxie.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from moxie._random import RandomState
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises


def supervised[E: SupervisedScorer](
    mut model: E, X: E.Input, y: E.Target,
) raises -> Float64 where conforms_to(E.SampleWeight, Deinitable):
    return model.fit(X, y).score(X, y)


def main() raises:
    var arrays = data()
    var X = arrays[0].copy()
    var y = arrays[1].copy()
    var params = grid([0.0, 4.0, 5.0, 9.0])
    var search = GridSearchCV[Adapter](DummyRegressor(), params, Adapter(), scoring=List[Scorer]([Scorer(0)]))
    comptime assert conforms_to(GridSearchCV[Adapter], RegressorMixin)
    assert_equal(supervised(search, X, y), -8.5)
    assert_true(search.__sklearn_is_fitted__())
    assert_equal(search.fit(X, y, None, None).predict(X).item(0), 4.0)
    var weights = Dense(Shape(10))
    for i in range(10):
        weights[Item(i)] = 1.0 if i < 5 else 10.0
    var expected = Scorer(0).score(search.best_estimator_(), X, y, weights.copy())
    assert_equal(search.score(X, y, metadata=weights.copy()), expected)
    with assert_raises():
        _ = search.score(X, y, sample_weight=weights.copy())
    var weighted = clone(search)
    _ = weighted.fit(X, y, weights.copy(), weights.copy())
    assert_equal(weighted.best_params_()["constant"], 5.0)
    comptime assert not conforms_to(GridSearchCV[Adapter], Copyable)
    var snapshot = clone(weighted)
    _ = snapshot.fit(X, y, weights.copy(), weights.copy())
    var changed = weighted.get_params()
    changed.space = ParameterGrid[Float64](grid([9.0]))
    assert_equal(weighted.set_params(changed).fit(X, y).predict(X).item(0), 9.0)
    assert_equal(snapshot.predict(X).item(0), 5.0)
    assert_false(clone(snapshot).__sklearn_is_fitted__())
    var results = snapshot.cv_results_()
    results.mean_test_score[Item(0, 0)] = 999.0
    assert_true(snapshot.cv_results_().mean_test_score.item(0, 0) != 999.0)
    var random = RandomizedSearchCV[Adapter](DummyRegressor(), ParameterGrid[Float64](params), Adapter(), scoring=List[Scorer]([Scorer(0)]), n_iter=4, random_state=7)
    assert_equal(supervised(random, X, y), -8.5)
    assert_equal(random.fit(X, y, None, None).predict(X).item(0), 4.0)
    comptime assert not conforms_to(RandomizedSearchCV[Adapter], Copyable)
    var random_clone = clone(random)
    assert_false(clone(random).__sklearn_is_fitted__())
    _ = random.set_params(n_iter=1)
    assert_false(random_clone.__sklearn_is_fitted__())
    assert_false(random.__sklearn_is_fitted__())
    var stream = RandomizedSearchCV[Adapter](DummyRegressor(), ParameterGrid[Float64](params), Adapter(), scoring=List[Scorer]([Scorer(0)]), n_iter=4)
    var rng = RandomState(7)
    assert_equal(stream.fit(X, y, None, None, rng).predict(X).item(0), 4.0)
    print("PASS fluent search, typed metadata, copy and clone")
