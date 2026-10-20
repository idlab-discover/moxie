from moxie._random import RandomState
from moxie.neural_network._plumbing import LayerLayout
from moxie.neural_network._mlp_engine import allocate, initialize, pack


def emit[dtype: DType](activation: String, seed: Int, units: List[Int]) raises:
    var params = allocate[dtype](LayerLayout(units))
    var rng = RandomState(seed)
    initialize(params, rng, activation)
    var vector = pack(params)
    for i in range(vector.shape[0]):
        print(vector.unsafe_get(i))
    print(rng.random_sample())


def main() raises:
    for activation in [String("identity"), String("logistic"), String("tanh"), String("relu")]:
        for seed in [0, 42]:
            emit[DType.float32](activation, seed, [2, 3, 1])
            emit[DType.float64](activation, seed, [2, 3, 1])
            emit[DType.float32](activation, seed, [1, 1])
            emit[DType.float64](activation, seed, [1, 1])
