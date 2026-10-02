from numojo import NDArray, Shape
from moxie.preprocessing._dense_transforms import normalize

def main() raises:
    var unsupported = NDArray[DType.int32](Shape(1,2))
    _ = normalize(unsupported)
