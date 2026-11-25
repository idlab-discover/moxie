from moxie.dummy import DummyRegressor
from moxie.model_selection import (
    CrossValidator, DenseSupervisedAdapter, GridSearchCV, GroupKFold, IndexArray, KFold,
    LeaveOneGroupOut, ParameterGrid, RandomizedSearchCV, SearchResults, select_rows,
)
from numojo import NDArray, Shape, Item, array, zeros
from std.testing import assert_equal, assert_false, assert_raises

comptime Model = DummyRegressor[]
comptime Adapter = DenseSupervisedAdapter[Model]
comptime Dense = NDArray[DType.float64]


@fieldwise_init
struct TargetCV(CrossValidator):
    comptime Input = Int
    comptime Target = Dense
    comptime Groups = IndexArray
    comptime Splits = KFold.Splits
    var reported_count: Int

    def get_n_splits(
        self, X: Optional[Int] = None, y: Optional[Dense] = None,
        groups: Optional[IndexArray] = None,
    ) raises -> Int:
        if not y:
            raise Error("expected search target")
        return self.reported_count

    def split(
        self, X: Int, y: Optional[Dense] = None, groups: Optional[IndexArray] = None,
    ) raises -> Self.Splits:
        if not y or y.value().size != X:
            raise Error("expected search target for current rows")
        return KFold(2 if y.value()[Item(0)] == 0.0 else 3).split(X)


def configure(mut model: Model, params: Dict[String, Float64]) raises:
    var config = model.get_params()
    config.constant = array[DType.float64]([params["constant"]], [1])
    _ = model.set_params(config)


def emit(case_name: String, results: SearchResults[Float64], best: Int, prediction: Float64) raises:
    print(case_name, "best", best, prediction)
    for candidate in range(len(results.params)):
        print(case_name, "candidate", results.params[candidate]["constant"],
              results.rank_test_score[Item(candidate, 0)])
        for fold in range(results.scores.test_scores.shape[1]):
            print(case_name, "test", results.scores.test_scores[Item(candidate, fold, 0)])
            print(case_name, "train", results.scores.train_scores.value()[Item(candidate, fold, 0)])


def main() raises:
    var X = zeros[DType.float64](Shape(12, 1))
    var y = Dense(Shape(12))
    var groups = IndexArray(Shape(12))
    var weights = Dense(Shape(12))
    for i in range(12):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64((i * 5) % 13)
        groups[Item(i)] = Int64(i % 4)
        weights[Item(i)] = Float64(i + 1)

    var grid = Dict[String, List[Float64]]()
    grid["constant"] = [2.0, 5.0, 8.0, 11.0]
    var model = Model(strategy="constant", constant=2.0)
    var adapter = Adapter(configure)
    var group_cv = GroupKFold(3)
    var search = GridSearchCV[Adapter](model, grid, adapter, return_train_score=True)
    assert_equal(search.fit(X, y, cv=group_cv, groups=groups.copy()).n_splits_(), 3)
    emit("grid", search.cv_results_(), search.best_index_(), search.predict(X)[Item(0)])
    with assert_raises(contains="groups are required"):
        _ = search.fit(X, y, cv=group_cv)
    assert_false(search.__sklearn_is_fitted__())
    with assert_raises(contains="groups must be rank one with matching length"):
        _ = search.fit(X, y, cv=group_cv, groups=IndexArray(Shape(2)))
    _ = search.fit(X, y)
    assert_equal(search.n_splits_(), 5)
    assert_equal(search.fit(X, y, cv=TargetCV(2)).n_splits_(), 2)
    var split_target = y.copy()
    split_target[Item(0)] = 1.0
    assert_equal(search.fit(X, y, cv=TargetCV(3), split_target=split_target^).n_splits_(), 3)
    with assert_raises(contains="inconsistent counts"):
        _ = search.fit(X, y, cv=TargetCV(3))
    assert_false(search.__sklearn_is_fitted__())

    var outer_cv = KFold(3).split(12)
    var first = outer_cv.next()
    var outer = first.take()
    var X_train = select_rows(X, outer.train)
    var y_train = select_rows(y, outer.train)
    var groups_train = select_rows(groups, outer.train)
    var weights_train = select_rows(weights, outer.train)
    var random = RandomizedSearchCV[Adapter](
        model, ParameterGrid[Float64](grid), adapter, n_iter=3, random_state=42,
        return_train_score=True,
    )
    assert_equal(random.fit(
        X_train, y_train, weights_train.copy(), weights_train^,
        cv=LeaveOneGroupOut(), groups=groups_train^,
    ).n_splits_(), 4)
    emit("random", random.cv_results_(), random.best_index_(), random.predict(X)[Item(0)])
