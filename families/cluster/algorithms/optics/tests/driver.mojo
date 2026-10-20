import numojo as nm
from std.sys import argv
from moxie.cluster import OPTICS
from moxie.cluster._optics import OPTICSSize

def run[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var d = Int(args[3])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n*d): x.unsafe_set(i, Float64(args[14+i]).cast[dtype]())
    var minimum = OPTICSSize(Int(Float64(args[4])))
    if Float64(args[4]) <= 1: minimum = OPTICSSize(Float64(args[4]))
    var size = Optional[OPTICSSize](None)
    if Float64(args[10]) >= 0:
        if Float64(args[10]) <= 1: size = OPTICSSize(Float64(args[10]))
        else: size = OPTICSSize(Int(Float64(args[10])))
    var eps = Optional[Float64](None)
    if Float64(args[8]) >= 0: eps = Float64(args[8])
    var model = OPTICS[dtype](min_samples=minimum, max_eps=Float64(args[5]), metric=args[6],
        cluster_method=args[7], eps=eps, xi=Float64(args[9]), min_cluster_size=size,
        predecessor_correction=Int(args[11]) != 0, algorithm=args[12], p=Float64(args[13]))
    _ = model.fit(x)
    for i in range(n):
        print("LABEL", model.labels_.item(i))
        print("ORDER", model.ordering_.item(i))
        print("CORE", model.core_distances_.item(i))
        print("REACH", model.reachability_.item(i))
        print("PRED", model.predecessor_.item(i))
    if model.cluster_hierarchy_:
        ref h = model.cluster_hierarchy_.value()
        print("SHAPE", h.shape[0], h.shape[1])
        for i in range(h.size): print("HIER", h.unsafe_get(i))

def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
