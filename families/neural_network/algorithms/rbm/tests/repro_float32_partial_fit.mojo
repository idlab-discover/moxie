from numojo import zeros, Shape
from moxie.neural_network import BernoulliRBM


def main() raises:
    var model = BernoulliRBM[DType.float32](1, batch_size=1, random_state=42)
    var X = zeros[DType.float32](Shape(1, 1))
    model.partial_fit(X)
