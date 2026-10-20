from numojo import zeros, Shape
from moxie.neural_network._plumbing import validate_matrix

def main() raises:
    var X = zeros[DType.int64](Shape(1, 1))
    validate_matrix(X)
