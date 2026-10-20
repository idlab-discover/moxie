"""Fit ordinary and nonnegative least squares, then predict and score."""
import numojo as nm
from numojo import Shape
from moxie.linear_model import LinearRegression


def main() raises:
    # y = 1 * x0 + 2 * x1 + 3
    var X = nm.zeros[DType.float64](Shape(4, 2))
    var y = nm.zeros[DType.float64](Shape(4))
    var rows: List[Float64] = [1, 1, 1, 2, 2, 2, 2, 3]
    for i in range(8):
        X.unsafe_set(i, rows[i])
    for i in range(4):
        y.unsafe_set(i, rows[2 * i] + 2 * rows[2 * i + 1] + 3)
    var model = LinearRegression()
    model.fit(X, y)
    print("coef_:", model.coef_.unsafe_get(0), model.coef_.unsafe_get(1))
    print("intercept_:", model.intercept_.unsafe_get(0), "rank_:", model.rank_.value())
    print("R2:", model.score(X, y))
    var test = nm.zeros[DType.float64](Shape(1, 2))
    test.unsafe_set(0, 3)
    test.unsafe_set(1, 5)
    print("predict([[3, 5]]):", model.predict(test).unsafe_get(0))
    var positive = LinearRegression(positive=True)
    positive.fit(X, y)
    print("positive coef_:", positive.coef_.unsafe_get(0), positive.coef_.unsafe_get(1))
