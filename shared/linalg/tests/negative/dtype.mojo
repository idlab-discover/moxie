from moxie._linalg import covariance_inverse
from numojo import Shape, zeros

def main() raises:
    var unsupported = zeros[DType.int64](Shape(2, 2))
    _ = covariance_inverse(unsupported)
