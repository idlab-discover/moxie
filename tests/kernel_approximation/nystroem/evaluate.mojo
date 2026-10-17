"""Parity driver: file `bits kernel gamma coef0 degree m seed n p X... q Y...` ('none' = None).

Prints component indices, normalization_, transform(Y) and the next RNG draw.
"""
from std.sys import argv
from numojo import NDArray, Shape
from moxie.kernel_approximation import Nystroem, NoCallableKernel
from moxie._random import RandomState
from moxie._kernel_nystroem import _workers


def option(token: String) raises -> Optional[Float64]:
    if token == "none":
        return None
    return Float64(token)


def evaluate[dtype: DType, output_dtype: DType = dtype](t: List[String]) raises:
    var n = Int(t[7])
    var p = Int(t[8])
    var x = NDArray[dtype](Shape(n, p))
    for k in range(n * p):
        x.unsafe_set(k, Scalar[dtype](Float64(t[9 + k])))
    var q = Int(t[9 + n * p])
    var y = NDArray[dtype](Shape(q, p))
    for k in range(q * p):
        y.unsafe_set(k, Scalar[dtype](Float64(t[10 + n * p + k])))
    var model = Nystroem[dtype, NoCallableKernel, output_dtype](t[1], gamma=option(t[2]), coef0=option(t[3]), degree=option(t[4]), n_components=Int(t[5]), n_jobs=Int(t[len(t)-1]))
    var rng = RandomState(Int(t[6]))
    model.fit_with_rng(x, rng)
    for k in range(model.component_indices_.size):
        print(model.component_indices_.unsafe_get(k))
    for k in range(model.normalization_.size):
        print(Float64(model.normalization_.unsafe_get(k)))
    var z = model.transform(y)
    for k in range(z.size):
        print(Float64(z.unsafe_get(k)))
    print(rng.random_sample())


def main() raises:
    var t = List[String]()
    with open(String(argv()[1]), "r") as f:
        for token in f.read().split():
            t.append(String(token))
    if t[0] == "32":
        if t[1] == "laplacian" and _workers(Optional[Int](Int(t[len(t)-1])), 2) == 1:
            evaluate[DType.float32, DType.float64](t)
        else:
            evaluate[DType.float32](t)
    else:
        evaluate[DType.float64](t)
