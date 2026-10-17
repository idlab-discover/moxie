from std.sys import argv
from numojo import NDArray, Shape
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler
from moxie._random import RandomState

def emit[dtype: DType](x: NDArray[dtype]):
    for i in range(x.size):
        print(Float64(x.unsafe_get(i)))

def evaluate[dtype: DType]() raises:
    var a = argv()
    var n = Int(a[3])
    var p = Int(a[4])
    var m = Int(a[5])
    var seed = Int(a[6])
    var parameter = Float64(a[7])
    var x = NDArray[dtype](Shape(n, p))
    for k in range(n*p):
        x.unsafe_set(k, Scalar[dtype](Float64(a[9+k])))
    var rng = RandomState(seed)
    if a[1] == "rbf":
        var model = RBFSampler[dtype](gamma=parameter, n_components=m)
        if a[8] == "scale":
            model.params.gamma_mode = "scale"
        model.fit_with_rng(x, rng)
        emit(model.random_weights_)
        emit(model.random_offset_)
        emit(model.transform(x))
        print(model._gamma)
    else:
        var model = SkewedChi2Sampler[dtype](skewedness=parameter, n_components=m)
        model.fit_with_rng(x, rng)
        emit(model.random_weights_)
        emit(model.random_offset_)
        emit(model.transform(x))
        print(0.0)
    print(rng.random_sample())
    print(rng.standard_normal())

def main() raises:
    var a = argv()
    if a[2] == "32":
        evaluate[DType.float32]()
    else:
        evaluate[DType.float64]()
