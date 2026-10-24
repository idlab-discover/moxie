from numojo import zeros, Shape
from moxie.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from std.testing import assert_raises
from kernel_dump import emit


def check[dtype: DType](n: Int, d: Int, h: Int, batch: Int) raises:
    var X = zeros[dtype](Shape(n, d))
    for i in range(X.size):
        X.unsafe_set(i, SIMD[dtype, 1](i % 7) / 6)
    var model = BernoulliRBM[dtype](h, batch_size=batch, n_iter=3, random_state=42)
    for repeat in range(2):
        var transformed = model.fit(X).transform(X)
        emit("weights", model.components_)
        emit("hidden_bias", model.intercept_hidden_)
        emit("visible_bias", model.intercept_visible_)
        emit("particles", model.h_samples_)
        emit("transform", transformed)
        emit("scores", model.score_samples(X))
    emit("gibbs", model.gibbs(X))
    comptime if dtype == DType.float64:
        var online = BernoulliRBM[dtype](h, batch_size=batch, random_state=42)
        for step in range(2):
            online.partial_fit(X)
            emit("online_weights", online.components_)
            emit("online_transform", online.transform(X))


def main() raises:
    with assert_raises(contains="NotImplementedError: MLPClassifier is not implemented"):
        var classifier = MLPClassifier()
    with assert_raises(contains="NotImplementedError: MLPRegressor is not implemented"):
        var regressor = MLPRegressor()
    check[DType.float64](1, 1, 1, 1)
    check[DType.float64](7, 3, 2, 4)
    check[DType.float64](2, 10, 5, 8)
    check[DType.float32](1, 1, 1, 1)
    check[DType.float32](7, 3, 2, 4)
    check[DType.float32](2, 10, 5, 8)
