# Gallery scenario adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
from moxie.model_selection import (
    DenseSupervisedAdapter, IndexArray, IndexSplit, PermutationCV, SplitLabels,
    StratifiedKFold, permutation_test_score,
)
from moxie.svm import SVC
from numojo import Item, NDArray, Shape
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_false

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Int]
comptime Classifier = SVC[Int]
comptime SVCAdapter = DenseSupervisedAdapter[Classifier, Labels]
comptime Permutations = 1000


struct ShuffledStratifiedCV(PermutationCV):
    comptime Adapter = SVCAdapter
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def split(
        mut self, X: Dense, y: Labels, groups: Optional[SplitLabels],
    ) raises -> List[IndexSplit]:
        self.calls += 1
        assert_false(Bool(groups))
        var labels = IndexArray(Shape(len(y)))
        for index in range(len(y)):
            labels[Item(index)] = Int64(y[index])
        var cursor = StratifiedKFold(2, shuffle=True, random_state=0).split(
            X.shape[0], y=labels^,
        )
        var folds = List[IndexSplit](capacity=2)
        while True:
            var pair = cursor.next()
            if not pair:
                return folds^
            folds.append(pair.take())


def run(path: String) raises:
    var text = open(path, "r").read()
    var lines = text.split("\n")
    var header = String(lines[0])
    var dimensions = header.split(" ")
    var rows = Int(String(dimensions[0]))
    var columns = Int(String(dimensions[1]))
    var X = Dense(Shape(rows, columns))
    var y = Labels(capacity=rows)
    for row in range(rows):
        var line = String(lines[row + 1])
        var fields = line.split(" ")
        y.append(Int(String(fields[0])))
        for column in range(columns):
            X[Item(row, column)] = bitcast[DType.float64](
                Int64(Int(String(fields[column + 1]))),
            )

    var model = Classifier(kernel="linear", random_state=7)
    var adapter = SVCAdapter()
    var cv = ShuffledStratifiedCV()
    var result = permutation_test_score[ShuffledStratifiedCV](
        model, X, y, adapter, adapter.default_scorer(), cv, None, None,
        n_permutations=Permutations, random_state=0,
    )
    assert_equal(cv.calls, Permutations + 1)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(result.permutation_scores.size, Permutations)
    print(result.score)
    for index in range(Permutations):
        print(result.permutation_scores[Item(index)])
    print(result.pvalue)


def main() raises:
    var args = argv()
    for index in range(1, len(args)):
        run(String(args[index]))
