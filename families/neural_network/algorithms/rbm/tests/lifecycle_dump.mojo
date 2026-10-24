"""Public BernoulliRBM lifecycle sequences compared by lifecycle_parity.py.

Covers binary data with more hidden than visible units, partial_fit and Gibbs
after fit, an explicit shared RNG context (sklearn's RandomState instance),
verbose training through that context, n_iter=0, set_params then refit and a
negative-stride input view.
"""
from numojo import NDArray, NDArrayStrides, zeros, Shape
from moxie._random import RandomState
from moxie.neural_network import BernoulliRBM
from kernel_dump import emit


def binary[dtype: DType](n: Int, d: Int) raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(n, d))
    for i in range(n):
        for j in range(d):
            X.unsafe_set(i * d + j, SIMD[dtype, 1](Int((i * 5 + j * 3 + i * j) % 7 < 3)))
    return X^


def state[dtype: DType](model: BernoulliRBM[dtype], X: NDArray[dtype]) raises:
    emit("weights", model.components_)
    emit("hidden_bias", model.intercept_hidden_)
    emit("visible_bias", model.intercept_visible_)
    emit("particles", model.h_samples_)
    emit("transform", model.transform(X))


def check[dtype: DType]() raises:
    var X = binary[dtype](23, 12)
    # Integer seed: fit, score, then partial_fit and Gibbs share one lazily created stream.
    var model = BernoulliRBM[dtype](16, learning_rate=0.05, batch_size=5, n_iter=4, random_state=3)
    _ = model.fit(X)
    state(model, X)
    emit("scores", model.score_samples(X))
    emit("gibbs", model.gibbs(X))
    comptime if dtype == DType.float64:
        model.partial_fit(X)
        state(model, X)
        emit("gibbs", model.gibbs(X))
    # Explicit context, the native counterpart of random_state=np.random.RandomState(11).
    var rng = RandomState(11)
    var shared = BernoulliRBM[dtype](6, batch_size=4, n_iter=3)
    _ = shared.fit(X, rng)
    state(shared, X)
    emit("scores", shared.score_samples(X, rng))
    emit("gibbs", shared.gibbs(X, rng))
    emit("scores", shared.score_samples(X, rng))
    comptime if dtype == DType.float64:
        shared.partial_fit(X, rng)
        state(shared, X)
    # Verbose training scores through the same context, which changes later draws.
    var noisy_rng = RandomState(5)
    var noisy = BernoulliRBM[dtype](4, batch_size=6, n_iter=2, verbose=1)
    _ = noisy.fit(X, noisy_rng)
    state(noisy, X)
    # n_iter=0 keeps the initial normal draws.
    var idle = BernoulliRBM[dtype](3, n_iter=0, random_state=8)
    _ = idle.fit(X)
    state(idle, X)
    # set_params then refit replaces every learned array.
    var params = model.get_params()
    params.n_components = 4
    params.learning_rate = 0.2
    params.batch_size = 7
    _ = model.set_params(params).fit(X)
    state(model, X)
    # Reversed-column view: arbitrary strides read the same logical matrix.
    var view = X.view_with_layout(Shape(23, 12), NDArrayStrides(12, -1), 11)
    var strided = BernoulliRBM[dtype](5, batch_size=10, n_iter=2, random_state=1)
    _ = strided.fit(view)
    state(strided, view)
    emit("scores", strided.score_samples(view))


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
