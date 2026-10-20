"""File-driven NCA comparison bridge; no Python in library execution."""
import numojo as nm
from std.sys import argv
from moxie.neighbors import NeighborhoodComponentsAnalysis, NCACallback
from moxie.neighbors._nca_math import initialize_spectral, loss_gradient
from moxie._random import RandomState


struct Trace(NCACallback):
    def __init__(out self):
        pass

    def __call__(mut self, x: nm.NDArray[DType.float64], n_iter: Int) raises:
        print("CI", n_iter)
        for i in range(x.size):
            print("C", x.unsafe_get(i))


def run[dtype: DType](args: List[String]) raises:
    var tokens = List[String]()
    with open(args[1], "r") as file:
        var contents = file.read()
        for token in contents.split():
            tokens.append(String(token))
    var n = Int(tokens[0])
    var d = Int(tokens[1])
    var k = Int(args[4])
    var X = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n * d):
        X.unsafe_set(i, Float64(tokens[2 + i]).cast[dtype]())
    var y = nm.zeros[dtype](nm.Shape(n))
    var codes = List[Int]()
    var count = 0
    for i in range(n):
        var label = Int(tokens[2 + n * d + i])
        y.unsafe_set(i, Float64(label).cast[dtype]())
        codes.append(label)
        count = max(count, label + 1)
    var A = nm.zeros[DType.float64](nm.Shape(k, d))
    for i in range(k * d):
        A.unsafe_set(i, Float64(tokens[2 + n * d + n + i]))
    var data = X.astype[DType.float64]()
    if args[3] == "gradient":
        var evaluated = loss_gradient(A, data, codes)
        print("L", evaluated[0])
        for i in range(A.size):
            print("G", evaluated[1].unsafe_get(i))
        return
    if args[3] == "pca_init" or args[3] == "lda_init":
        var initial = initialize_spectral(
            data, codes, count, k, args[3] == "lda_init"
        )
        print("S", initial.shape[0], initial.shape[1])
        for i in range(initial.size):
            print("A", initial.unsafe_get(i))
        return
    var model = NeighborhoodComponentsAnalysis[dtype, Trace](
        k,
        init=args[3],
        random_state=42,
        max_iter=Int(args[5]),
        tol=Float64(args[6]),
    )
    if args[3] == "array":
        model.set_params(init=A)
    model.fit(X, y)
    var output = model.transform(X)
    print("S", model.components_.shape[0], model.n_features_in_)
    print("I", model.n_iter_)
    for i in range(model.components_.size):
        print("A", model.components_.unsafe_get(i))
    for i in range(output.size):
        print("T", output.unsafe_get(i))


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if args[2] == "f32":
        run[DType.float32](args)
    else:
        run[DType.float64](args)
