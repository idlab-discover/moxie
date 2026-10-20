import numojo as nm
from std.sys import argv
from std.math import min
from moxie.cluster import MiniBatchKMeans
from moxie.cluster._kmeans import KMeansInit, KMeansNInit
from moxie._random import RandomState
from moxie.cluster._centroid_selection import count_order


def snapshot[dtype: DType](model: MiniBatchKMeans[dtype], rng: RandomState) raises:
    print("META", model.n_steps_, model.n_iter_.value() if model.n_iter_ else -1, model.inertia_.value() if model.inertia_ else -1, model._no_improvement, model._n_since_last_reassign, model._ewa_inertia.value() if model._ewa_inertia else -1, model._init_size, model._batch_size)
    for i in range(model.cluster_centers_.size): print("CENTER", model.cluster_centers_.item(i))
    for i in range(model._counts.size): print("COUNT", model._counts.item(i))
    for i in range(model.labels_.size): print("LABEL", model.labels_.item(i))
    print("RNGPOS", rng._pos)
    print("RNGGAUSS", Int(rng._has_gauss), rng._gauss)
    for word in rng._key: print("RNGKEY", word)
    print("END")


def run[dtype: DType]() raises:
    var args = argv()
    var mode = args[2]
    var n = Int(args[3])
    var d = Int(args[4])
    var k = Int(args[5])
    var seed = Int(args[6])
    var batch = Int(args[7])
    var maximum = Int(args[8])
    var tolerance = Float64(args[9])
    var runs = Int(args[10])
    var init_mode = args[11]
    var init_size = Optional[Int]()
    if Int(args[12]): init_size = Int(args[12])
    var patience = Optional[Int]()
    if Int(args[13]) >= 0: patience = Int(args[13])
    var ratio = Float64(args[14])
    var labels = Bool(Int(args[15]))
    var x = nm.zeros[dtype](nm.Shape(n, d))
    var w = nm.zeros[dtype](nm.Shape(n))
    for i in range(n * d): x.unsafe_set(i, Float64(args[16 + i]).cast[dtype]())
    for i in range(n): w.unsafe_set(i, Float64(args[16 + n * d + i]).cast[dtype]())
    if mode == "sort":
        var order = count_order(w)
        for i in range(order.size): print("ORDER", order.item(i))
        return
    var init = KMeansInit[dtype](init_mode)
    if init_mode == "array":
        var centers = nm.zeros[dtype](nm.Shape(k, d))
        for i in range(k * d): centers.unsafe_set(i, Float64(args[16 + n * d + n + i]).cast[dtype]())
        init = KMeansInit[dtype](centers)
    var n_init = KMeansNInit()
    if runs: n_init = KMeansNInit(runs)
    var model = MiniBatchKMeans[dtype](k, init=init^, n_init=n_init^, max_iter=maximum, batch_size=batch, tol=tolerance, init_size=init_size, max_no_improvement=patience, reassignment_ratio=ratio, compute_labels=labels, random_state=seed)
    var rng = RandomState(seed)
    if mode == "init":
        model._validate_params(x)
        model._configure(x)
        _ = rng.choice(n, model._init_size)
        for _ in range(model._n_init):
            model.cluster_centers_ = model._initialize(x, w, rng)
            snapshot(model, rng)
    elif mode == "fit":
        _ = model.fit(x, random_state=rng, sample_weight=w.view())
        snapshot(model, rng)
    elif mode == "prefix":
        # Refit from the same RNG for each prefix to inspect every fit update.
        var steps = maximum * n // min(batch, n)
        for step in range(1, steps + 1):
            var local = RandomState(seed)
            model.params.max_iter = step
            # Prefixes use n == effective batch; each epoch is one step.
            _ = model.fit(x, random_state=local, sample_weight=w.view())
            snapshot(model, local)
    else:
        for step in range(maximum):
            var size = n if step == 0 else 1 + (step * 7) % n
            var data = nm.zeros[dtype](nm.Shape(size, d))
            var weights = nm.zeros[dtype](nm.Shape(size))
            for i in range(size):
                var row = (i + step * 3) % n
                for f in range(d): data.unsafe_set(i * d + f, x.item(row, f))
                weights.unsafe_set(i, w.item(row))
            if mode == "internal":
                _ = model.partial_fit(data, sample_weight=weights.view())
                snapshot(model, model._random_state.value())
            else:
                _ = model.partial_fit(data, random_state=rng, sample_weight=weights.view())
                snapshot(model, rng)


def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
