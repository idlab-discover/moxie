"""Dense SelectKBest and SVC with a fixed-width list target and fold-local fit."""

from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.feature_selection import BuiltinScorer, SelectKBest
from moxie.model_selection import DenseSupervisedAdapter, cross_val_score, cross_validate
from moxie.model_selection._split import _rows
from moxie.pipeline import Pipeline, make_pipeline
from moxie.svm import SVC

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]
comptime Scorer = BuiltinScorer[target_type=Labels]
comptime Model = Pipeline[SelectKBest[Scorer], SVC[]]
comptime Adapter = DenseSupervisedAdapter[Model, Labels]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(40, 4))
    var y = Labels()
    for i in range(40):
        var group = i % 2
        X[Item(i, 0)] = Float64((i * 7) % 17) / 4 + Float64(group)
        X[Item(i, 1)] = Float64((i * 11) % 13) / 4 + Float64(group * Int(i < 25))
        X[Item(i, 2)] = Float64((i * 13) % 19) / 4 + Float64(2 * group * Int(i >= 25))
        X[Item(i, 3)] = Float64((i * 3) % 11) / 4
        y.append(9.0 if group == 0 else -3.0)
    return (X^, y^)


def main() raises:
    var data = dataset()
    ref X = data[0]
    ref y = data[1]
    var pipe = make_pipeline(SelectKBest(Scorer(), k=1), SVC())
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict(X)
    with assert_raises(contains="inconsistent sample counts"):
        _ = pipe.fit(X, [9.0])
    assert_false(pipe.__sklearn_is_fitted__())

    var cv = cross_val_score[Adapter](pipe, X, y, Adapter())
    var validation = cross_validate[Adapter](pipe, X, y, Adapter(),
                                             return_estimator=True, return_indices=True)
    assert_false(pipe.__sklearn_is_fitted__())
    assert_equal(cv.size, 5)
    ref folds = validation.indices.value()
    ref fitted = validation.estimators.value()
    for fold in range(cv.size):
        assert_equal(cv.item(fold), validation.scores.test_scores.item(0, fold, 0))
        var selected = fitted[fold].transformer.get_support(indices=True)
        assert_equal(len(selected[List[Int]]), 1)
        print("fold", cv.item(fold), selected[List[Int]][0])
        var predictions = fitted[fold].predict(_rows(X, folds[fold].test))
        for row in range(folds[fold].test.size):
            print("prediction", folds[fold].test.item(row), predictions[row])

    _ = pipe.fit(X, y)
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    var selected = pipe.transformer.get_support(indices=True)
    print("full_support", selected[List[Int]][0])

    var params = pipe.get_params()
    params.transformer.param = 2
    _ = pipe.set_params(params).fit(X, y)
    selected = pipe.transformer.get_support(indices=True)
    assert_equal(len(selected[List[Int]]), 2)
    for feature in selected[List[Int]]:
        print("refit_support", feature)
    print("PASS: SelectKBest and SVC under cross_val_score")
