import numojo as nm
from std.sys import argv
from std.memory import bitcast
from moxie.cluster import DBSCAN
from moxie.cluster._dbscan import _core_weight_sum


def run[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var d = Int(args[3])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n*d): x.unsafe_set(i, Float64(args[10+i]).cast[dtype]())
    var weights = Optional[nm.NDArray[dtype]](None)
    if args[9] == "weighted":
        var w = nm.zeros[dtype](nm.Shape(n))
        for i in range(n): w.unsafe_set(i, Float64(args[10+n*d+i]).cast[dtype]())
        weights = w^
    var model = DBSCAN[dtype](Float64(args[4]), min_samples=Int(args[5]), algorithm=args[6], metric=args[7], leaf_size=2, p=Float64(args[8]))
    _ = model.fit(x, sample_weight=weights)
    for i in range(n): print("LABEL", model.labels_.item(i))
    for i in range(model.core_sample_indices_.size): print("CORE", model.core_sample_indices_.item(i))
    for i in range(model.components_.size): print("COMPONENT", model.components_.unsafe_get(i))
    print("SHAPE", model.components_.shape[0], model.components_.shape[1])


def reduction[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var w = nm.zeros[dtype](nm.Shape(n))
    var indices = List[Int]()
    for i in range(n):
        w.unsafe_set(i, Float64(args[3+i]).cast[dtype]())
        indices.append(i)
    var total = _core_weight_sum(w, indices, 0, n)
    comptime if dtype == DType.float32: print(bitcast[DType.uint32](total))
    else: print(bitcast[DType.uint64](total))


def main() raises:
    if argv()[1] == "sum32":
        reduction[DType.float32]()
        return
    if argv()[1] == "sum64":
        reduction[DType.float64]()
        return
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
