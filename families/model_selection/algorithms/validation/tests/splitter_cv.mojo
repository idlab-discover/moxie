from moxie.dummy import DummyRegressor
from moxie.model_selection import (
    CVResult, DenseSupervisedAdapter, GroupKFold, IndexArray, StratifiedKFold,
    cross_val_score, cross_validate,
)
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_false, assert_raises

comptime Model = DummyRegressor[]
comptime Dense = NDArray[DType.float64]
comptime Adapter = DenseSupervisedAdapter[Model]


def emit(case_name: String, result: CVResult[Model]) raises:
    ref folds = result.indices.value()
    assert_equal(len(result.estimators.value()), len(folds))
    for k in range(len(folds)):
        print(case_name, k, "test", result.scores.test_scores[Item(0, k, 0)])
        print(case_name, k, "train", result.scores.train_scores.value()[Item(0, k, 0)])
        for i in range(folds[k].train.size):
            print(case_name, k, "train_index", folds[k].train[Item(i)])
        for i in range(folds[k].test.size):
            print(case_name, k, "test_index", folds[k].test[Item(i)])


def main() raises:
    var X = Dense(Shape(12, 1))
    var y = Dense(Shape(12))
    var labels = IndexArray(Shape(12))
    var groups = IndexArray(Shape(12))
    var weights = Dense(Shape(12))
    for i in range(12):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64((i * 5) % 13)
        labels[Item(i)] = Int64(i % 2)
        groups[Item(i)] = Int64(i % 4)
        weights[Item(i)] = Float64(i + 1)

    var model = Model()
    var adapter = Adapter()
    var cv = GroupKFold(3)
    var grouped = cross_validate[Adapter](
        model, X, y, adapter, cv=cv, groups=groups.copy(),
        return_train_score=True, return_indices=True, return_estimator=True,
        fit_metadata=weights.copy(), score_metadata=None,
    )
    emit("grouped", grouped)
    var scores = cross_val_score[Adapter](
        model, X, y, adapter, cv=cv, groups=groups.copy(),
        fit_metadata=weights^, score_metadata=None,
    )
    for k in range(scores.size):
        print("grouped", k, "score", scores[Item(k)])

    var stratified = StratifiedKFold(3, shuffle=True, random_state=42)
    var mixed = cross_validate[Adapter](
        model, X, y, adapter, cv=stratified, split_target=labels.copy(),
        return_train_score=True, return_indices=True, return_estimator=True,
    )
    emit("stratified", mixed)
    var stratified_scores = cross_val_score[Adapter](
        model, X, y, adapter, cv=stratified, split_target=labels^,
    )
    for k in range(stratified_scores.size):
        print("stratified", k, "score", stratified_scores[Item(k)])

    with assert_raises(contains="groups are required"):
        _ = cross_val_score[Adapter](model, X, y, adapter, cv=cv)
    with assert_raises(contains="groups must be rank one with matching length"):
        _ = cross_validate[Adapter](
            model, X, y, adapter, cv=cv, groups=IndexArray(Shape(2)),
        )
    assert_false(model.__sklearn_is_fitted__())
