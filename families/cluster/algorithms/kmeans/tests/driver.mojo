import numojo as nm
from std.sys import argv
from moxie.cluster import KMeans, kmeans_plusplus, k_means
from moxie.cluster._kmeans import KMeansInit, KMeansNInit
from moxie._random import RandomState


def run[dtype: DType]() raises:
    var args = argv()
    var mode = args[2]
    var n = Int(args[3])
    var d = Int(args[4])
    var k = Int(args[5])
    var seed = Int(args[6])
    var trials = Int(args[7])
    var maximum = Int(args[8])
    var tolerance = Float64(args[9])
    var runs = Int(args[10])
    var init_mode = args[11]
    var supplied_norms = Int(args[12])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    var w = nm.zeros[dtype](nm.Shape(n))
    for i in range(n * d): x.unsafe_set(i, Float64(args[13 + i]).cast[dtype]())
    for i in range(n): w.unsafe_set(i, Float64(args[13 + n * d + i]).cast[dtype]())
    var rng = RandomState(seed)
    if mode == "seed":
        var norms = Optional[nm.NDArray[dtype]]()
        if supplied_norms:
            var values = nm.zeros[dtype](nm.Shape(n))
            for i in range(n): values.unsafe_set(i, Float64(args[13 + n * d + n + i]).cast[dtype]())
            norms = values^
        var local = Optional[Int]()
        if trials: local = trials
        var result = kmeans_plusplus(x, k, random_state=rng, sample_weight=w.view(), x_squared_norms=norms, n_local_trials=local)
        for i in range(k):
            print("INDEX", result.indices.item(i))
            for j in range(d): print("CENTER", result.centers.item(i, j))
    else:
        var init = KMeansInit[dtype](init_mode)
        if init_mode == "array":
            var centers = nm.zeros[dtype](nm.Shape(k, d))
            for i in range(k * d): centers.unsafe_set(i, Float64(args[13 + n * d + n + i]).cast[dtype]())
            init = KMeansInit[dtype](centers)
        var n_init = KMeansNInit()
        if runs: n_init = KMeansNInit(runs)
        var model = KMeans[dtype](k, init=init^, n_init=n_init^, max_iter=maximum, tol=tolerance, algorithm=mode)
        _ = model.fit(x, random_state=rng, sample_weight=w.view())
        print("META", model.n_iter_, model.inertia_, model.score(x, sample_weight=w.view()))
        var predicted = model.predict(x)
        var transformed = model.transform(x)
        for i in range(n):
            print("LABEL", model.labels_.item(i))
            print("PREDICT", predicted.item(i))
            for j in range(k): print("TRANSFORM", transformed.item(i, j))
        for i in range(k):
            for j in range(d): print("CENTER", model.cluster_centers_.item(i, j))
    print("RNG", rng.next_uint32())


def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
