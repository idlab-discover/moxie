"""Wrap a binary classifier as a three-class OneVsRestClassifier.

The component is the deterministic test classifier from tests/ovr_fixtures.mojo;
any binary estimator with NDArray[float64] features and NDArray[int64] 0/1
targets works the same way.
"""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import clone
from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import MarginProbaClassifier


def main() raises:
    var X = nm.zeros[DType.float64](Shape(6, 2))
    var rows: List[List[Float64]] = [[10, 10], [8, 10], [-5, 5.5], [-5.4, 5.5], [-20, -20], [-15, -20]]
    for i in range(6):
        for j in range(2):
            X[Item(i, j)] = rows[i][j]
    var y = nm.zeros[DType.int64](Shape(6))
    var codes: List[Int] = [0, 0, 1, 1, 2, 2]
    for i in range(6):
        y[Item(i)] = Int64(codes[i])

    var model = OneVsRestClassifier(MarginProbaClassifier())
    model.fit(X, y)
    print("classes_:", model.classes_)
    print("estimators_:", len(model.estimators_))

    var queries = nm.zeros[DType.float64](Shape(3, 2))
    var points: List[List[Float64]] = [[-19, -20], [9, 9], [-5, 5]]
    for i in range(3):
        for j in range(2):
            queries[Item(i, j)] = points[i][j]
    print("predict:", model.predict(queries))
    print("predict_proba:", model.predict_proba(queries))

    # clone copies configuration only; the copy must be fitted again.
    var fresh = clone(model)
    print("clone fitted:", fresh.__sklearn_is_fitted__())
