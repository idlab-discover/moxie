from std.sys import argv
from numojo import NDArray, zeros, Shape
from moxie.manifold import trustworthiness


def read_matrix[dtype: DType](path: String) raises -> NDArray[dtype]:
    var text: String
    with open(path, "r") as handle:
        text = handle.read()
    var tokens = text.split()
    var rows = Int(String(tokens[0]))
    var columns = Int(String(tokens[1]))
    var x = zeros[dtype](Shape(rows, columns))
    for i in range(rows * columns):
        x.unsafe_set(i, Float64(String(tokens[i + 2])).cast[dtype]())
    return x^


def run[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var y = read_matrix[dtype](String(args[3]))
    print(trustworthiness(x, y, n_neighbors=Int(String(args[4])), metric=String(args[5])))


def main() raises:
    var args = argv()
    if args[1] == "float32":
        run[DType.float32]()
    else:
        run[DType.float64]()
