"""Prepare independent MLP-shaped storage; this example performs no training."""
from numojo import NDArray, Shape, zeros
from moxie.neural_network._plumbing import LayerLayout, LayerTensors
from std.testing import assert_equal


def main() raises:
    var layout = LayerLayout([2, 3, 1])
    var coefficients = List[NDArray[DType.float64]]()
    var biases = List[NDArray[DType.float64]]()
    coefficients.append(zeros[DType.float64](Shape(2, 3)))
    coefficients.append(zeros[DType.float64](Shape(3, 1)))
    biases.append(zeros[DType.float64](Shape(3)))
    biases.append(zeros[DType.float64](Shape(1)))
    var snapshot = LayerTensors[DType.float64](layout, coefficients, biases)
    coefficients[0].unsafe_set(0, 7)
    assert_equal(snapshot.coefs[0].unsafe_get(0), 0)
    print("Independent storage for a 2 -> 3 -> 1 network; no engine implemented")
