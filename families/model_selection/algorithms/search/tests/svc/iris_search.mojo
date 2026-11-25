# Gallery scenario adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, IndexArray, IndexSplit, KFold,
    ParameterGrid, RandomizedSearchCV, SearchAdapter, SearchResults,
    cross_val_score, select_rows,
)
from moxie.svm import Gamma, SVC
from numojo import Item, NDArray, Shape
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_false

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Int]
comptime Classifier = SVC[Int]
comptime DenseSVC = DenseSupervisedAdapter[Classifier, Labels]


def folds(n: Int, seed: Int) raises -> List[IndexSplit]:
    var cursor = KFold(4, shuffle=True, random_state=seed).split(n)
    var splits = List[IndexSplit](capacity=4)
    while True:
        var pair = cursor.next()
        if not pair:
            return splits^
        splits.append(pair.take())


@fieldwise_init
struct ShuffledSVC(SearchAdapter):
    comptime Scorer = DenseSVC.Scorer
    comptime Value = Float64

    var seed: Int

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        return DenseSVC().n_samples(X, y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        return DenseSVC().metadata_rows(metadata, rows)

    def fit(self, mut model: Classifier, X: Dense, y: Labels, metadata: Optional[Dense]) raises:
        DenseSVC().fit(model, X, y, metadata)

    def configure(self, mut model: Classifier, params: Dict[String, Float64]) raises:
        var config = model.get_params()
        for key in params.keys():
            if key == "C":
                config.C = params[key]
            elif key == "gamma":
                config.gamma = Gamma(params[key])
            else:
                raise Error("ValueError: unknown SVC parameter " + key)
        _ = model.set_params(config)

    def default_scorer(self) raises -> Self.Scorer:
        return DenseSVC().default_scorer()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return folds(self.n_samples(X, y), self.seed)


comptime Grid = GridSearchCV[ShuffledSVC]
comptime Random = RandomizedSearchCV[ShuffledSVC]
comptime OuterCV = DenseSupervisedAdapter[Grid, Labels]


def load_iris(path: String) raises -> Tuple[Dense, Labels]:
    var text = open(path, "r").read()
    var lines = text.split("\n")
    var X = Dense(Shape(150, 4))
    var y = Labels(capacity=150)
    for row in range(X.shape[0]):
        var line = String(lines[row])
        var fields = line.split(" ")
        y.append(Int(String(fields[0])))
        for column in range(X.shape[1]):
            X[Item(row, column)] = bitcast[DType.float64](
                Int64(Int(String(fields[column + 1]))),
            )
    return (X^, y^)


def emit_results(
    results: SearchResults[Float64], index: Int, score: Float64,
    best: Dict[String, Float64], name: String, trial: Int,
) raises:
    print(name, trial, "best", index, score, best["C"], best["gamma"])
    assert_equal(results.scores.test_scores.shape[1], 4)
    ref train = results.scores.train_scores.value()
    ref mean_train = results.mean_train_score.value()
    ref std_train = results.std_train_score.value()
    for candidate in range(len(results.params)):
        ref params = results.params[candidate]
        var metric = Item(candidate, 0)
        print(name, trial, "candidate", params["C"], params["gamma"], results.rank_test_score[metric])
        print(name, trial, "summary", results.mean_test_score[metric], results.std_test_score[metric],
              mean_train[metric], std_train[metric])
        for fold in range(4):
            print(name, trial, "test", results.scores.test_scores[Item(candidate, fold, 0)])
            print(name, trial, "train", train[Item(candidate, fold, 0)])


def emit_refit(labels: Labels, decision: Dense, support: NDArray[DType.int32], name: String) raises:
    for label in labels:
        print(name, 0, "predict", label)
    assert_equal(decision.shape[0], len(labels))
    assert_equal(decision.shape[1], 3)
    for row in range(decision.shape[0]):
        for column in range(decision.shape[1]):
            print(name, 0, "decision", decision[Item(row, column)])
    for i in range(support.size):
        print(name, 0, "support", support[Item(i)])


def main() raises:
    var dataset = load_iris(String(argv()[1]))
    ref X = dataset[0]
    ref y = dataset[1]
    var grid = Dict[String, List[Float64]]()
    grid["C"] = [1.0, 10.0, 100.0]
    grid["gamma"] = [0.01, 0.1]
    var model = Classifier(kernel="rbf")
    for trial in range(30):
        var search = Grid(
            model, grid, ShuffledSVC(trial), cv=folds(len(y), trial), return_train_score=True,
        )
        _ = search.fit(X, y)
        emit_results(search.cv_results_(), search.best_index_(), search.best_score_(),
                     search.best_params_(), "grid", trial)
        if trial == 0:
            emit_refit(search.predict(X), search.decision_function(X),
                       search.best_estimator_().support_, "grid")
        var nested = Grid(model, grid, ShuffledSVC(trial))
        var scores = cross_val_score[OuterCV](nested, X, y, OuterCV(), cv=folds(len(y), trial))
        assert_false(nested.__sklearn_is_fitted__())
        for i in range(scores.size):
            print("grid", trial, "nested", scores[Item(i)])
    var random = Random(
        model, ParameterGrid[Float64](grid), ShuffledSVC(0),
        n_iter=4, random_state=0, cv=folds(len(y), 0), return_train_score=True,
    )
    _ = random.fit(X, y)
    emit_results(random.cv_results_(), random.best_index_(), random.best_score_(),
                 random.best_params_(), "random", 0)
    emit_refit(random.predict(X), random.decision_function(X),
               random.best_estimator_().support_, "random")
    assert_false(model.__sklearn_is_fitted__())
