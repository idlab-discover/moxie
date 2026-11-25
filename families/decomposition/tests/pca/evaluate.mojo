"""Native command-line driver for differential PCA validation."""
from std.sys import argv
from numojo import NDArray, Shape, zeros
from moxie.decomposition import PCA
from moxie.decomposition._pca import ComponentCount, IteratedPower
from moxie.decomposition_serialization import PCASchema
from moxie.serialization import dumps, loads


def emit[dtype: DType](name: String, x: NDArray[dtype]) raises:
    print(name + "_dtype", String(dtype))
    if x.size == 0: return
    var flat = x.flatten()
    for i in range(flat.size): print(name, Float64(flat.unsafe_get(i)))


def evaluate[dtype: DType]() raises:
    var args = argv()
    var m = Int(args[2])
    var p = Int(args[3])
    var components = ComponentCount(NoneType())
    if args[4] == "int": components = Int(args[5])
    elif args[4] == "fraction": components = Float64(args[5])
    elif args[4] == "mle": components = String("mle")
    var power = IteratedPower(String("auto"))
    if Int(args[10]) >= 0: power = Int(args[10])
    var model = PCA[dtype](components, svd_solver=args[6], whiten=Bool(Int(args[7])),
        copy=Bool(Int(args[8])), iterated_power=power, n_oversamples=Int(args[11]),
        power_iteration_normalizer=args[12], random_state=Int(args[13]))
    var x = zeros[dtype](Shape(m, p))
    var mode = args[14]
    for i in range(m * p): x.unsafe_set(i, SIMD[dtype, 1](Float64(args[15 + i])))
    var original = x.copy()
    if mode == "fit":
        _ = model.fit(x)
    elif mode == "fit_transform":
        emit("fit_transform", model.fit_transform(x))
    elif mode == "inplace":
        emit("fit_transform", model.fit_transform_inplace(x))
    else:
        raise Error("Unrecognized PCA fit mode")
    model = PCASchema[dtype].decode(loads(dumps(PCASchema[dtype].encode(model))))
    print("k", model.n_components_)
    print("solver", model._fit_svd_solver)
    print("noise", model.noise_variance_)
    emit("components", model.components_.value())
    emit("mean", model.mean_.value())
    emit("variance", model.explained_variance_.value())
    emit("ratio", model.explained_variance_ratio_.value())
    emit("singular", model.singular_values_.value())
    var projected = model.transform(original)
    emit("transform", projected)
    emit("inverse", model.inverse_transform(projected))
    emit("covariance", model.get_covariance())
    emit("input", x)
    if args[9] == "score":
        emit("precision", model.get_precision())
        emit("score_samples", model.score_samples(original))
        print("score", model.score(original))


def main() raises:
    var args = argv()
    if args[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
