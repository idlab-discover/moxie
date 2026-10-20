"""Initialize an MLP-shaped parameter set; this example does not train."""
from moxie._random import RandomState
from moxie.neural_network._plumbing import LayerLayout
from moxie.neural_network._mlp_engine import allocate, initialize, pack


def main() raises:
    var layout = LayerLayout([2, 3, 1])
    var params = allocate[DType.float64](layout)
    var rng = RandomState(42)
    initialize(params, rng, "relu")
    var vector = pack(params)
    print("Initialized", vector.shape[0], "parameters")
