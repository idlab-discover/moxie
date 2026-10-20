"""Public family imports from the assembled namespace: pixi run start."""
from numojo import Shape, zeros
from moxie.linear_model import LinearRegression, Ridge, RidgeClassifier, ElasticNet, Lasso


def main() raises:
    var X = zeros[DType.float64](Shape(4, 1))
    var y = zeros[DType.float64](Shape(4))
    var labels = zeros[DType.int64](Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, 2.0 * Float64(i) + 1.0)
        labels.unsafe_set(i, Int64(i >= 2))
    var ols = LinearRegression()
    ols.fit(X, y)
    var ridge = Ridge(alpha=0.1)
    ridge.fit(X, y, sample_weight=2.0)
    var classifier = RidgeClassifier()
    classifier.fit(X, labels)
    print("LinearRegression R2:", ols.score(X, y))
    print("Ridge R2:", ridge.score(X, y))
    print("RidgeClassifier accuracy:", classifier.score(X, labels))
    var elastic = ElasticNet(alpha=0.1)
    elastic.fit(X, y)
    var lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)
    print("ElasticNet R2:", elastic.score(X, y))
    print("Lasso R2:", lasso.score(X, y))
