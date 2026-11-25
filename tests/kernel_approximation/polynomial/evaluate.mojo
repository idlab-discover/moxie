"""Print hashes, map and next draw. Pass a second argument for Float32 output."""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.kernel_approximation import PolynomialCountSketch
from moxie._random import check_random_state, seed_global_random_state


def evaluate[dtype: DType, output_dtype: DType = DType.float64](t: List[String]) raises:
    var n = Int(t[6])
    var p = Int(t[7])
    var x = NDArray[dtype](Shape(n, p))
    for k in range(n * p):
        x.unsafe_set(k, Scalar[dtype](Float64(t[8 + k])))
    var model = PolynomialCountSketch[dtype, output_dtype=output_dtype](
        gamma=Float64(t[1]), degree=Int(t[2]), coef0=Float64(t[3]), n_components=Int(t[4])
    )
    seed_global_random_state(Int(t[5]))
    _ = model.fit(x)
    for k in range(model.indexHash_.size):
        print(model.indexHash_.unsafe_get(k))
    for k in range(model.bitHash_.size):
        print(model.bitHash_.unsafe_get(k))
    var z = model.transform(x)
    for k in range(z.size):
        print(Float64(z.unsafe_get(k)))
    with check_random_state() as rng:
        print(rng[].random_sample())


def main() raises:
    var t = List[String]()
    with open(String(argv()[1]), "r") as f:
        for token in f.read().split():
            t.append(String(token))
    if len(argv()) > 2:
        if t[0] == "32":
            evaluate[DType.float32, DType.float32](t)
        else:
            evaluate[DType.float64, DType.float32](t)
    elif t[0] == "32":
        evaluate[DType.float32](t)
    else:
        evaluate[DType.float64](t)
