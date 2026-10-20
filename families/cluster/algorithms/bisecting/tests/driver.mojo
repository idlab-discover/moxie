import numojo as nm
from std.sys import argv
from moxie.cluster import BisectingKMeans
from moxie._random import RandomState


def run[dtype: DType]() raises:
    var a = argv()
    var n = Int(a[2])
    var d = Int(a[3])
    var q = Int(a[4])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    var w = nm.zeros[dtype](nm.Shape(n))
    var query = nm.zeros[dtype](nm.Shape(q, d))
    for i in range(n*d): x.unsafe_set(i, Float64(a[13+i]).cast[dtype]())
    for i in range(n): w.unsafe_set(i, Float64(a[13+n*d+i]).cast[dtype]())
    for i in range(q*d): query.unsafe_set(i, Float64(a[13+n*d+n+i]).cast[dtype]())
    var rng = RandomState(Int(a[6]))
    var model = BisectingKMeans[dtype](Int(a[5]), init=a[7], n_init=Int(a[8]), max_iter=Int(a[9]), tol=Float64(a[10]), algorithm=a[11], bisecting_strategy=a[12])
    _ = model.fit(x, random_state=rng, sample_weight=w.view())
    print("INERTIA", model.inertia_)
    print("SCORE", model.score(x, sample_weight=w.view()))
    for i in range(n): print("LABEL", model.labels_.item(i))
    for i in range(model.cluster_centers_.size): print("CENTER", model.cluster_centers_.unsafe_get(i))
    var predicted = model.predict(query)
    var transformed = model.transform(query)
    for i in range(q): print("PREDICT", predicted.item(i))
    for i in range(transformed.size): print("DISTANCE", transformed.unsafe_get(i))
    for i in range(len(model._nodes)):
        print("NODE", model._nodes[i].left, model._nodes[i].right, model._nodes[i].label, model._nodes[i].score, model._nodes[i].weight, model._nodes[i].inertia)
        for f in range(d): print("NODECENTER", model._nodes[i].center.item(0, f))
    print("RNG", rng.next_uint32())


def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
