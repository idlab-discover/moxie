from search_support import Adapter, Scorer, Dense, data, grid
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.base import clone
from moxie.model_selection import validation_curve, GridSearchCV, RandomizedSearchCV, ParameterGrid, DiscreteDistribution, ParameterDomain, SearchRefit, SearchRefitPolicy, SearchResults, ErrorPolicy
from moxie._random import RandomState
from numojo import Item, Shape
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

struct LastCandidate(SearchRefitPolicy):
    comptime Value = Float64
    comptime Results = SearchResults[Float64]
    def __init__(out self):
        pass
    def select(self, results: Self.Results) raises -> Int:
        return len(results.params) - 1

struct InvalidCandidate(SearchRefitPolicy):
    comptime Value = Float64
    comptime Results = SearchResults[Float64]
    def __init__(out self):
        pass
    def select(self, results: Self.Results) raises -> Int:
        return len(results.params)


def main() raises:
    var arrays = data()
    var model = DummyRegressor()
    _ = model.fit(arrays[0], arrays[1])
    var search = GridSearchCV[Adapter](model, grid([0.0, 4.0, 5.0, 9.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]), return_train_score=True)
    with assert_raises():
        _ = search.predict(arrays[0])
    search.fit(arrays[0], arrays[1], None, None)
    assert_equal(search.best_index_(), 1)
    assert_equal(search.best_params_()["constant"], 4.0)
    assert_equal(search.best_score_(), -8.5)
    assert_equal(search.n_splits_(), 5)
    assert_false(search.multimetric_())
    assert_equal(search.best_estimator_().n_features_in_.value(), 2)
    assert_equal(search.predict(arrays[0])[Item(0)], 4.0)
    assert_equal(model.predict(arrays[0])[Item(0)], 4.5)
    assert_equal(search.score(arrays[0], arrays[1], None), -8.5)
    assert_true(search.refit_time_() >= 0)
    var result = search.cv_results_()
    # The same adapter configures both curves and search through ParameterAdapter.
    var folds = Adapter().default_cv(arrays[0], arrays[1])
    var curve = validation_curve[Adapter](model, arrays[0], arrays[1], Adapter(), Scorer(0), folds, None, None, param_name="constant", param_range=List[Float64]([0.0, 4.0, 5.0, 9.0]))
    for candidate in range(4):
        for fold in range(5):
            assert_equal(curve[0][Item(candidate, fold)], result.scores.train_scores.value()[Item(candidate, fold, 0)])
            assert_equal(curve[1][Item(candidate, fold)], result.scores.test_scores[Item(candidate, fold, 0)])

    assert_equal(result.rank_test_score[Item(1, 0)], 1)
    assert_equal(result.rank_test_score[Item(2, 0)], 1)
    assert_equal(result.rank_test_score[Item(0, 0)], 3)
    assert_equal(result.mean_train_score.value().shape[0], 4)
    result.mean_test_score[Item(0, 0)] = 999
    assert_true(search.cv_results_().mean_test_score[Item(0, 0)] != 999)
    var cloned = clone(search)
    assert_false(cloned.__sklearn_is_fitted__())
    cloned.fit(arrays[0], arrays[1], None, None)
    assert_equal(cloned.best_index_(), 1)
    var params = cloned.get_params()
    params.space = ParameterGrid[Float64](grid([9.0]))
    _ = cloned.set_params(params)
    assert_false(cloned.__sklearn_is_fitted__())
    cloned.fit(arrays[0], arrays[1], None, None)
    assert_equal(cloned.best_params_()["constant"], 9.0)
    var no_refit = GridSearchCV[Adapter](model, grid([4.0, 5.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]), refit=SearchRefit[Float64](False))
    no_refit.fit(arrays[0], arrays[1], None, None)
    assert_equal(no_refit.best_index_(), 0)
    assert_equal(no_refit.best_score_(), -8.5)
    with assert_raises():
        _ = no_refit.predict(arrays[0])
    var scorers: List[Scorer] = [Scorer(0), Scorer(1)]
    var names: List[String] = ["mse", "mae"]
    var multi = GridSearchCV[Adapter](model, grid([0.0, 4.0, 5.0]), Adapter(), scoring=scorers.copy(), metric_names=names.copy(), refit=SearchRefit[Float64]("mae"))
    multi.fit(arrays[0], arrays[1], None, None)
    assert_true(multi.multimetric_())
    assert_equal(multi.best_score_(), -2.5)
    var none_multi = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), scoring=scorers.copy(), metric_names=names.copy(), refit=SearchRefit[Float64](False))
    none_multi.fit(arrays[0], arrays[1], None, None)
    with assert_raises():
        _ = none_multi.best_index_()
    var custom = GridSearchCV[Adapter, LastCandidate](model, grid([4.0, 9.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]))
    custom.fit(arrays[0], arrays[1], None, None)
    assert_equal(custom.best_index_(), 1)
    with assert_raises():
        _ = custom.best_score_()
    var invalid = GridSearchCV[Adapter, InvalidCandidate](model, grid([4.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]))
    with assert_raises():
        invalid.fit(arrays[0], arrays[1], None, None)
    assert_false(invalid.__sklearn_is_fitted__())
    var spaces = List[Dict[String, List[Float64]]]()
    var bad = Dict[String, List[Float64]]()
    bad["invalid"] = [1.0]
    spaces.append(bad.copy())
    spaces.append(grid([4.0]))
    var failures = GridSearchCV[Adapter](model, spaces, Adapter(), scoring=List[Scorer]([Scorer(0)]))
    failures.fit(arrays[0], arrays[1], None, None)
    assert_equal(failures.best_index_(), 1)
    var failed_result = failures.cv_results_()
    assert_true(isnan(failed_result.mean_test_score[Item(0, 0)]))
    assert_true(failed_result.status[0].failed)
    assert_equal(failed_result.status[0].score_time, 0.0)
    assert_false(Bool(failed_result.parameter("constant")[0]))
    assert_equal(failed_result.parameter("constant")[1].value(), 4.0)
    var failed_params = failures.get_params()
    failed_params.space = ParameterGrid[Float64](bad)
    _ = failures.set_params(failed_params)
    with assert_raises():
        failures.fit(arrays[0], arrays[1], None, None)
    assert_false(failures.__sklearn_is_fitted__())
    with assert_raises():
        _ = failures.best_estimator_()
    var refit_fail = GridSearchCV[Adapter](model, grid([99.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]), error_score=ErrorPolicy(False, 1000))
    with assert_raises():
        refit_fail.fit(arrays[0], arrays[1], None, None)
    assert_false(refit_fail.__sklearn_is_fitted__())
    var nan_search = GridSearchCV[Adapter](model, grid([0.0, 4.0]), Adapter(), scoring=List[Scorer]([Scorer(3)]))
    nan_search.fit(arrays[0], arrays[1], None, None)
    assert_equal(nan_search.best_index_(), 0)
    assert_equal(nan_search.cv_results_().rank_test_score[Item(1, 0)], 1)
    var random = RandomizedSearchCV[Adapter](model, ParameterGrid[Float64](grid([0.0, 2.0, 4.0, 5.0, 9.0])), Adapter(), scoring=List[Scorer]([Scorer(0)]), n_iter=3, random_state=42)
    random.fit(arrays[0], arrays[1], None, None)
    var before = random.cv_results_()
    random.fit(arrays[0], arrays[1], None, None)
    var after = random.cv_results_()
    for i in range(3):
        assert_equal(before.params[i]["constant"], after.params[i]["constant"])
    var random_clone = clone(random)
    assert_false(random_clone.__sklearn_is_fitted__())
    random_clone.fit(arrays[0], arrays[1], None, None)
    assert_equal(random_clone.best_score_(), random.best_score_())
    var stream = RandomizedSearchCV[Adapter](model, ParameterGrid[Float64](grid([0.0, 2.0, 4.0, 5.0, 9.0])), Adapter(), scoring=List[Scorer]([Scorer(0)]), n_iter=3)
    var rng = RandomState(42)
    stream.fit(arrays[0], arrays[1], None, None, rng)
    assert_equal(stream.best_score_(), random.best_score_())
    print("PASS grid/random search lifecycle, real estimator, ties, failures, refit and seeded repeat fits")
