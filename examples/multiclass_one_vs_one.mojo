"""Wrap a binary classifier as a four-class OneVsOneClassifier.

The component is the deterministic test classifier from tests/ovr_fixtures.mojo;
any binary estimator with NDArray[float64] features, NDArray[int64] 0/1 targets
and NDArray[int64] predictions works the same way.
"""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import clone
from moxie.multiclass import OneVsOneClassifier
from ovr_fixtures import MarginClassifier


def main() raises:
    var X = nm.zeros[DType.float64](Shape(8, 2))
    var rows: List[List[Float64]] = [[10, 10], [8, 10], [-5, 5.5], [-5.4, 5.5],
                                     [-20, -20], [-15, -20], [10, -10], [8, -12]]
    for i in range(8):
        for j in range(2):
            X[Item(i, j)] = rows[i][j]
    var y = nm.zeros[DType.int64](Shape(8))
    var codes: List[Int] = [0, 0, 1, 1, 2, 2, 3, 3]
    for i in range(8):
        y[Item(i)] = Int64(codes[i])

    var model = OneVsOneClassifier(MarginClassifier())
    model.fit(X, y)
    print("classes_:", model.classes_)
    # One component per class pair: 4 * 3 / 2.
    print("estimators_:", len(model.estimators_))

    var queries = nm.zeros[DType.float64](Shape(3, 2))
    var points: List[List[Float64]] = [[-19, -20], [9, 9], [9, -11]]
    for i in range(3):
        for j in range(2):
            queries[Item(i, j)] = points[i][j]
    print("predict:", model.predict(queries))
    # Votes per class plus a tie-breaking confidence term in (-1/3, 1/3).
    print("decision_function:", model.decision_function(queries))

    # clone copies configuration only; the copy must be fitted again.
    var fresh = clone(model)
    print("clone fitted:", fresh.__sklearn_is_fitted__())
