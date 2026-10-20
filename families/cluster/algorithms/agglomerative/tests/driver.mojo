import numojo as nm
from std.sys import argv
from moxie.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree, linkage_tree

def run[dtype: DType]() raises:
    var args = argv()
    var n = Int(args[3])
    var d = Int(args[4])
    var x = nm.zeros[dtype](nm.Shape(n, d))
    for i in range(n*d): x.unsafe_set(i, Float64(args[9+i]).cast[dtype]())
    var k = Optional[Int](None)
    var threshold = Optional[Float64](None)
    if Int(args[7]) > 0: k = Int(args[7])
    else: threshold = Float64(args[8])
    if args[2] == "feature":
        var model = FeatureAgglomeration[dtype](k, linkage=args[5], metric=args[6], distance_threshold=threshold, compute_distances=True)
        var transformed = model.fit_transform(x)
        var inverse = model.inverse_transform(transformed)
        for i in range(model.labels_.size): print("LABEL", model.labels_.item(i))
        for i in range(model.children_.size): print("CHILD", model.children_.unsafe_get(i))
        for i in range(model.distances_.value().size): print("DIST", model.distances_.value().item(i))
        for i in range(transformed.size): print("TRANSFORM", transformed.unsafe_get(i))
        for i in range(inverse.size): print("INVERSE", inverse.unsafe_get(i))
        print("COUNT", model.n_clusters_, model.n_leaves_, model.n_connected_components_, model.n_features_in_)
    elif args[2] == "tree" or args[2] == "vector":
        if args[2] == "vector": x = x.reshape(nm.Shape(n*d))
        var tree = ward_tree(x, n_clusters=k, return_distance=True) if args[5] == "ward" else linkage_tree(x, n_clusters=k, linkage=args[5], affinity=args[6], return_distance=True)
        tree.validate()
        for i in range(tree.children.size): print("CHILD", tree.children.unsafe_get(i))
        for i in range(tree.distances.value().size): print("DIST", tree.distances.value().item(i))
        print("COUNT", tree.n_leaves, tree.n_connected_components, Int(Bool(tree.parents)))
    else:
        var model = AgglomerativeClustering[dtype](k, linkage=args[5], metric=args[6], distance_threshold=threshold, compute_distances=True)
        var labels = model.fit_predict(x)
        for i in range(labels.size): print("LABEL", labels.item(i))
        for i in range(model.children_.size): print("CHILD", model.children_.unsafe_get(i))
        for i in range(model.distances_.value().size): print("DIST", model.distances_.value().item(i))
        print("COUNT", model.n_clusters_, model.n_leaves_, model.n_connected_components_, model.n_features_in_)

def main() raises:
    if argv()[1] == "32": run[DType.float32]()
    else: run[DType.float64]()
