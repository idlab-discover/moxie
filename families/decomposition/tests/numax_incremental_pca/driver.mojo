from std.sys import argv
from numojo import Shape, zeros
from moxie.decomposition import IncrementalPCA
from moxie_numax import NumaxIncrementalPCABackend
from decomposition_testing import emit


def evaluate_ipca[dtype: DType, p: Int]() raises:
    var args = argv()
    var n = Int(args[2])
    var k: Optional[Int] = None
    if Int(args[4]) > 0: k = Int(args[4])
    var batch_size: Optional[Int] = None
    if Int(args[5]) > 0: batch_size = Int(args[5])
    var model = IncrementalPCA[dtype, NumaxIncrementalPCABackend[40, p, p]](k,
        batch_size=batch_size, whiten=Bool(Int(args[6])), copy=Bool(Int(args[7])))
    var x = zeros[dtype](Shape(n, p))
    for i in range(n * p): x.unsafe_set(i, Scalar[dtype](Float64(args[9 + i])))
    var original = x.copy()
    if args[8] == "fit":
        var fitted = model.fit_transform_inplace(x)
        emit("fit_transform", fitted)
    else:
        var step = Int(args[5])
        var start = 0
        while start < n:
            var stop = min(n, start + step)
            var batch = zeros[dtype](Shape(stop - start, p))
            for i in range(stop - start):
                for j in range(p): batch.unsafe_set(i * p + j, x.unsafe_get((start + i) * p + j))
            model.partial_fit_inplace(batch)
            for i in range(stop - start):
                for j in range(p): x.unsafe_set((start + i) * p + j, batch.unsafe_get(i * p + j))
            start = stop
    print("k", model.n_components_)
    print("seen", model.n_samples_seen_)
    print("noise", model.noise_variance_)
    emit("mean", model.mean_.value())
    emit("var", model.var_.value())
    emit("components", model.components_.value())
    emit("singular", model.singular_values_.value())
    emit("variance", model.explained_variance_.value())
    emit("ratio", model.explained_variance_ratio_.value())
    var projected = model.transform(original)
    emit("transform", projected)
    emit("inverse", model.inverse_transform(projected))
    emit("covariance", model.get_covariance())
    emit("input", x)


def dispatch[dtype: DType]() raises:
    var p = Int(argv()[3])
    if p == 1: evaluate_ipca[dtype, 1]()
    elif p == 3: evaluate_ipca[dtype, 3]()
    elif p == 7: evaluate_ipca[dtype, 7]()
    else: raise Error("test driver shape unsupported")


def main() raises:
    if argv()[1] == "f32": dispatch[DType.float32]()
    else: dispatch[DType.float64]()
