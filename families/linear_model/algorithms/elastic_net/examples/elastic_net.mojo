from moxie.linear_model import ElasticNet, Lasso
from numojo import zeros, Shape

def main() raises:
    var X = zeros[DType.float64](Shape(3, 1))
    var y = zeros[DType.float64](Shape(3))
    for i in range(3):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Float64(2 * i + 1))
    var model = ElasticNet(alpha=0.1)
    model.fit(X, y)
    var lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)
    print("ElasticNet coef:", model.coef_.unsafe_get(0), "gap:", model.dual_gap_.unsafe_get(0))
    print("Lasso coef:", lasso.coef_.unsafe_get(0), "R2:", lasso.score(X, y))
