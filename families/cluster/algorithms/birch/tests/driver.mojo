import numojo as nm
from std.sys import argv
from moxie.cluster import Birch


def tree[dtype: DType](model: Birch[dtype], index: Int) raises:
    var node = model.get_node(index)
    print("NODE", Int(node.is_leaf), len(node.subclusters_))
    for cluster in node.subclusters_:
        print("SAMPLES", cluster.n_samples_)
        print("SS", cluster.squared_sum_)
        for j in range(cluster.linear_sum_.size): print("LS", cluster.linear_sum_.item(j))
        if cluster.child_: tree(model, cluster.child_.value())


def run[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[2])
    var d = Int(args[3])
    var threshold = Float64(args[4])
    var branching = Int(args[5])
    var k = Optional[Int](None)
    if Int(args[6]) > 0: k = Int(args[6])
    var batch_size = Int(args[7])
    var labels = Int(args[8]) == 1
    var final_k = Int(args[9])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n*d): x.unsafe_set(i, Float64(args[11+i]).cast[dtype]())
    var backing = nm.zeros[dtype](nm.Shape(n*2, d*2))
    if Int(args[10]) == 1:
        for i in range(n):
            for j in range(d): backing[nm.Item(n*2-1-i*2,j*2)] = x.item(i,j)
        x = backing.view_with_layout(nm.Shape(n,d), nm.Strides(-4*d,2), (n*2-1)*d*2)
    var model = Birch[dtype](threshold=threshold, branching_factor=branching, n_clusters=k, compute_labels=labels)
    if batch_size == 0:
        _ = model.fit(x)
    else:
        for start in range(0,n,batch_size):
            var count = min(batch_size,n-start)
            var batch = nm.zeros[dtype](nm.Shape(count,d))
            for i in range(count):
                for j in range(d): batch[nm.Item(i,j)] = x.item(start+i,j)
            _ = model.partial_fit(batch)
    if final_k >= 0:
        var final_count = Optional[Int](None)
        if final_k > 0: final_count = final_k
        _ = model.set_params(n_clusters=final_count).partial_fit()
    print("META", model.subcluster_centers_.shape[0], model.n_features_in_, Int(Bool(model.labels_)), Int(model.n_clusters_warning_))
    for i in range(model.subcluster_centers_.size): print("CENTER", model.subcluster_centers_.unsafe_get(i))
    for i in range(model.subcluster_labels_.size): print("SUBLABEL", model.subcluster_labels_.item(i))
    if model.labels_:
        for i in range(model.labels_.value().size): print("LABEL", model.labels_.value().item(i))
    var prediction = model.predict(x)
    var transformed = model.transform(x)
    for i in range(n): print("PREDICT", prediction.item(i))
    for i in range(transformed.size): print("TRANSFORM", transformed.unsafe_get(i))
    var query = nm.zeros[dtype](nm.Shape(n,d))
    for i in range(n):
        for j in range(d): query[nm.Item(i,j)] = x.item(i,j) + Scalar[dtype](0.137)
    prediction = model.predict(query)
    for i in range(n): print("QUERY", prediction.item(i))
    for name in model.get_feature_names_out(): print("NAME", name)
    tree(model, model.root_.value())


def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
