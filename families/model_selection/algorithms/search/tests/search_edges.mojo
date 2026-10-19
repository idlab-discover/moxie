from search_support import Adapter, Scorer, Dense, data, grid
from search import LastCandidate
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, SearchRefit, ErrorPolicy, IndexSplit
from moxie.model_selection._search import _ranks, _moments
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import isnan
from std.memory import bitcast
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.


def _indices(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = NDArray[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def main() raises:
    var arrays = data()
    var model = DummyRegressor()
    var weights = Dense(Shape(10))
    for i in range(10):
        weights[Item(i)] = Float64(i + 1)
    # Empty dictionary is the original model configuration, not an empty search.
    var empty_grid = Dict[String, List[Float64]]()
    var weighted = GridSearchCV[Adapter](model, empty_grid, Adapter())
    weighted.fit(arrays[0], arrays[1], weights.copy(), weights.copy())
    assert_equal(weighted.predict(arrays[0])[Item(0)], 6.0)
    assert_equal(weighted.best_index_(), 0)
    print("weighted_score", weighted.best_score_())
    assert_false(Bool(weighted.cv_results_().mean_train_score))
    # Explicit uneven folds and owned indices; CV accepts repeated/overlapping tests.
    var pairs = List[IndexSplit]()
    pairs.append(IndexSplit(_indices([3, 4, 5, 6, 7, 8, 9]), _indices([0, 1, 2])))
    pairs.append(IndexSplit(_indices([0, 1, 2, 3, 4, 5]), _indices([6, 7, 8, 9])))
    var search = GridSearchCV[Adapter](model, grid([3.0, 4.0]), Adapter(), scoring=List[Scorer]([Scorer(0)]), cv=pairs^, refit=False)
    search.fit(arrays[0], arrays[1], None, None)
    assert_equal(search.n_splits_(), 2)
    var result = search.cv_results_()
    for c in range(2):
        assert_equal(result.mean_test_score[Item(c, 0)], (result.scores.test_scores[Item(c, 0, 0)] + result.scores.test_scores[Item(c, 1, 0)]) / 2)
    # A scorer failure fills only its metric, preserving all successful fits.
    var failed_score = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), scoring=List[Scorer]([Scorer(0), Scorer(2)]), metric_names=List[String](["good", "bad"]), refit="good", error_score=ErrorPolicy(False, -12))
    failed_score.fit(arrays[0], arrays[1], None, None)
    var failures = failed_score.cv_results_()
    for f in range(5):
        assert_false(failures.status[f].failed)
        assert_equal(failures.scores.test_scores[Item(0, f, 1)], -12.0)
    var raising = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), scoring=List[Scorer]([Scorer(2)]), error_score=ErrorPolicy(True, 0))
    with assert_raises():
        raising.fit(arrays[0], arrays[1], None, None)
    # Default refit=True is invalid even for a one-key multi-metric dictionary.
    var bad_metric = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), metric_names=List[String](["only"]))
    with assert_raises():
        bad_metric.fit(arrays[0], arrays[1], None, None)
    var duplicate = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), scoring=List[Scorer]([Scorer(0), Scorer(1)]), metric_names=List[String](["same", "same"]), refit=False)
    with assert_raises():
        duplicate.fit(arrays[0], arrays[1], None, None)
    var zero_folds = GridSearchCV[Adapter](model, grid([4.0]), Adapter(), cv=List[IndexSplit]())
    with assert_raises():
        zero_folds.fit(arrays[0], arrays[1], None, None)
    var no_candidates = GridSearchCV[Adapter](model, List[Dict[String, List[Float64]]](), Adapter())
    with assert_raises():
        no_candidates.fit(arrays[0], arrays[1], None, None)
    var unknown = Dict[String, List[Float64]]()
    unknown["typo"] = [1.0]
    var wrong = GridSearchCV[Adapter](model, unknown, Adapter())
    with assert_raises():
        wrong.fit(arrays[0], arrays[1], None, None)
    # A failed repeated fit clears a previous winner even without set_params.
    var previous = GridSearchCV[Adapter](model, grid([4.0]), Adapter())
    previous.fit(arrays[0], arrays[1], None, None)
    var malformed = Dense(Shape(1))
    with assert_raises():
        previous.fit(arrays[0], malformed, None, None)
    assert_false(previous.__sklearn_is_fitted__())
    with assert_raises():
        _ = previous.cv_results_()
    var random = RandomizedSearchCV[Adapter](model, ParameterGrid[Float64](grid([4.0])), Adapter(), random_state=0)
    random.fit(arrays[0], arrays[1], None, None)
    assert_equal(len(random.cv_results_().params), 1)
    random.set_params(n_iter=0)
    with assert_raises():
        random.fit(arrays[0], arrays[1], None, None)
    assert_false(random.__sklearn_is_fitted__())
    # Custom multimetric selection intentionally has no implicit score name.
    var custom = GridSearchCV[Adapter, LastCandidate](model, grid([4.0]), Adapter(), scoring=List[Scorer]([Scorer(0), Scorer(1)]), metric_names=List[String](["mse", "mae"]))
    custom.fit(arrays[0], arrays[1], None, None)
    with assert_raises():
        _ = custom.score(arrays[0], arrays[1], None)
    assert_equal(custom.score_metric("mae", arrays[0], arrays[1], None), -2.5)
    print("PASS weighted fits, explicit folds, invalid configuration, scorer failures and stale-state rejection")
