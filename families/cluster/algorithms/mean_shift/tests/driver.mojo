import numojo as nm
from std.sys import argv
from moxie.cluster import MeanShift, estimate_bandwidth, get_bin_seeds

def run[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[3])
    var d = Int(args[4])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n*d): x.unsafe_set(i, Float64(args[10+i]).cast[dtype]())
    if args[2] == "bandwidth":
        print("BANDWIDTH", estimate_bandwidth(x, quantile=Float64(args[5]), n_samples=Int(args[6]) if Int(args[6]) > 0 else Optional[Int](None), random_state=Int(args[7])))
        return
    if args[2] == "bins":
        var bins = get_bin_seeds(x, Float64(args[5]), Int(args[6]))
        print("SHAPE", bins.shape[0], bins.shape[1])
        for i in range(bins.size): print("CENTER", bins.unsafe_get(i))
        return
    var seeds = Optional[nm.NDArray[dtype]](None)
    var count = Int(args[9])
    if count >= 0:
        var s = nm.zeros[dtype](nm.Shape(count, d))
        for i in range(count*d): s.unsafe_set(i, Float64(args[10+n*d+i]).cast[dtype]())
        seeds = s^
    var bandwidth = Optional[Float64](None)
    if Float64(args[5]) >= 0: bandwidth = Float64(args[5])
    var model = MeanShift[dtype](bandwidth=bandwidth, seeds=seeds.copy(), bin_seeding=Int(args[6]) != 0,
        min_bin_freq=Int(args[7]), max_iter=Int(args[8]), cluster_all=args[2] != "orphans")
    _ = model.fit(x)
    print("SHAPE", model.cluster_centers_.shape[0], d)
    print("ITER", model.n_iter_)
    for i in range(model.cluster_centers_.size): print("CENTER", model.cluster_centers_.unsafe_get(i))
    var prediction = model.predict(x)
    for i in range(n):
        print("LABEL", model.labels_.item(i))
        print("PREDICT", prediction.item(i))

def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
