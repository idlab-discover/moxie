"""Parity driver: file `bits gamma degree coef0 m seed n p values...`; prints hashes, map, next draw."""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.kernel_approximation import PolynomialCountSketch
from moxie._random import RandomState


def evaluate[dtype: DType](t: List[String]) raises:
    var n = Int(t[6])
    var p = Int(t[7])
    var x = NDArray[dtype](Shape(n, p))
    for k in range(n * p):
        x.unsafe_set(k, Scalar[dtype](Float64(t[8 + k])))
    var model = PolynomialCountSketch[dtype](gamma=Float64(t[1]), degree=Int(t[2]), coef0=Float64(t[3]), n_components=Int(t[4]))
    var rng = RandomState(Int(t[5]))
    model.fit_with_rng(x, rng)
    for k in range(model.indexHash_.size):
        print(model.indexHash_.unsafe_get(k))
    for k in range(model.bitHash_.size):
        print(model.bitHash_.unsafe_get(k))
    var z = model.transform(x)
    for k in range(z.size):
        print(z.unsafe_get(k))
    print(rng.random_sample())


def main() raises:
    var t = List[String]()
    with open(String(argv()[1]), "r") as f:
        for token in f.read().split():
            t.append(String(token))
    if t[0] == "32":
        evaluate[DType.float32](t)
    else:
        evaluate[DType.float64](t)
