from moxie._labels import encode_column
from numojo import Shape
import numojo as nm

def main() raises:
    var unsupported = nm.zeros[DType.float16](Shape(2))
    _ = encode_column(unsupported)
