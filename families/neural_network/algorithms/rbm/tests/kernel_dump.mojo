from numojo import NDArray, Shape, array, zeros
from moxie._random import RandomState
from moxie.neural_network._rbm import BernoulliRBM


def emit[dtype: DType](name: String, X: NDArray[dtype]):
    print(name, end="")
    for i in range(X.size):
        print(" ", Float64(X.unsafe_get(i)), sep="", end="")
    print()


def check[dtype: DType](n: Int, d: Int, h: Int, batch: Int, steps: Int) raises:
    var rng = RandomState(42)
    var state = BernoulliRBM[dtype](h, batch_size=batch)
    state.initialize(h, d, batch, rng)
    var X = zeros[dtype](Shape(n, d))
    for i in range(X.size):
        X.unsafe_set(i, SIMD[dtype, 1](i % 7) / 6)
    for _ in range(steps):
        state.update(X, 0.1, rng)
    emit("weights", state.components_)
    emit("hidden_bias", state.intercept_hidden_)
    emit("visible_bias", state.intercept_visible_)
    emit("particles", state.h_samples_)
    emit("probabilities", state.mean_hiddens(X))
    emit("energy", state.free_energy(X))
    emit("gibbs", state.gibbs(X, rng))
    emit("scores", state.score_samples(X, rng))
    print("rng", rng.state_position(), rng.has_gauss(), rng.cached_gauss())
    for i in range(624):
        print("word", rng.state_word(i))


def activation_tails() raises:
    var rng = RandomState(42)
    var state = BernoulliRBM[DType.float64](1, batch_size=1)
    state.initialize(1, 1, 1, rng)
    state.components_.unsafe_set(0, 1.0)
    var X = array[DType.float64](
        [-500.0, -60.0, -30.0, -3.0, -0.3, 0.0, 0.3, 3.0, 30.0, 60.0, 500.0],
        [11, 1],
    )
    emit("tail_probabilities", state.mean_hiddens(X))
    emit("tail_energy", state.free_energy(X))


def main() raises:
    check[DType.float64](1, 1, 1, 1, 0)
    check[DType.float64](7, 3, 2, 4, 5)
    check[DType.float64](2, 10, 5, 8, 3)
    check[DType.float32](1, 1, 1, 1, 0)
    check[DType.float32](7, 3, 2, 4, 5)
    check[DType.float32](2, 10, 5, 8, 3)
    activation_tails()
