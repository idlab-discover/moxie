"""Parity driver: read `dtype steps interval|none n p values...` from a file, print the map."""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.kernel_approximation import AdditiveChi2Sampler


def evaluate[dtype: DType](tokens: List[String]) raises:
    var steps = Int(tokens[1])
    var interval: Optional[Float64] = None
    if tokens[2] != "none":
        interval = Float64(tokens[2])
    var n = Int(tokens[3])
    var p = Int(tokens[4])
    var x = NDArray[dtype](Shape(n, p))
    for k in range(n * p):
        x.unsafe_set(k, Scalar[dtype](Float64(tokens[5 + k])))
    var model = AdditiveChi2Sampler[dtype](sample_steps=steps, sample_interval=interval)
    var z = model.fit_transform(x)
    for k in range(z.size):
        print(Float64(z.unsafe_get(k)))
    for name in model.get_feature_names_out():
        print(name)


def main() raises:
    var tokens = List[String]()
    with open(String(argv()[1]), "r") as f:
        for token in f.read().split():
            tokens.append(String(token))
    if tokens[0] == "32":
        evaluate[DType.float32](tokens)
    else:
        evaluate[DType.float64](tokens)
