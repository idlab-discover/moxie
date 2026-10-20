from moxie.cluster import Birch
from moxie.cluster import AgglomerativeClustering, FeatureAgglomeration, linkage_tree, ward_tree
from moxie.cluster import OPTICS, cluster_optics_dbscan
from moxie.cluster._optics import OPTICSSize
from moxie.cluster import MeanShift
from moxie.cluster import DBSCAN, dbscan
from moxie import ConsumerA, ConsumerB
from moxie._random import RandomState
from moxie.cluster import affinity_propagation, MiniBatchKMeans, BisectingKMeans
import numojo as nm
from std.testing import assert_equal

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(1, 2))
    x[nm.Item(0, 0)] = 3
    x[nm.Item(0, 1)] = 4
    var a = ConsumerA()
    var b = ConsumerB()
    var owned = a.snapshot(x)
    var view = owned.view()
    view[nm.Item(0, 0)] = 0
    var norms = b.norms(owned)
    assert_equal(norms.item(0), 16)
    assert_equal(x.item(0, 0), 3)
    var state = RandomState(42)
    var expected = RandomState(42)
    assert_equal(a.draw(state), expected.next_uint32())
    assert_equal(b.draw(state), expected.next_uint32())
    var features = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        features[nm.Item(i, 0)] = Float64(1 if i < 3 else 4)
        features[nm.Item(i, 1)] = Float64(2 if i % 3 == 0 else (4 if i % 3 == 1 else 0))
    var model = a.fit(features)
    var labels = b.predict(model, features)
    for i in range(6):
        assert_equal(labels.item(i), Int64(i // 3))
    var similarities = nm.zeros[DType.float64](nm.Shape(1, 1))
    var result = affinity_propagation(similarities, random_state=5, return_n_iter=True)
    assert_equal(result.labels.item(0), Int64(0))
    assert_equal(result.n_iter.value(), 0)
    print("isolated cluster consumer passed")
    var kmeans = a.fit_kmeans(features)
    var klabels = b.predict_kmeans(kmeans, features)
    for i in range(6): assert_equal(klabels.item(i), kmeans.labels_.item(i))
    var minibatch = MiniBatchKMeans(2, random_state=42)
    _ = minibatch.partial_fit(features)
    _ = minibatch.partial_fit(features)
    assert_equal(minibatch.n_steps_, 2)
    var minibatch_labels = minibatch.predict(features)
    for i in range(6): assert_equal(minibatch_labels.item(i), minibatch.labels_.item(i))

    var bisecting = BisectingKMeans(2, random_state=42)
    _ = bisecting.fit(features)
    var bisecting_labels = bisecting.predict(features)
    for i in range(6): assert_equal(bisecting_labels.item(i), bisecting.labels_.item(i))

    var density = DBSCAN(2, min_samples=2, algorithm="ball_tree")
    _ = density.fit(features)
    var density_result = dbscan(features, 2, min_samples=2, algorithm="kd_tree")
    for i in range(6): assert_equal(density.labels_.item(i), density_result.labels.item(i))

    var shifted = MeanShift(bandwidth=2.0)
    _ = shifted.fit(features)
    var shifted_labels = shifted.predict(features)
    for i in range(6): assert_equal(shifted_labels.item(i), shifted.labels_.item(i))

    var optics = OPTICS(min_samples=OPTICSSize(2), cluster_method="dbscan", eps=Float64(2))
    _ = optics.fit(features)
    var extracted = cluster_optics_dbscan(reachability=optics.reachability_, core_distances=optics.core_distances_, ordering=optics.ordering_, eps=2)
    for i in range(6): assert_equal(extracted.item(i), optics.labels_.item(i))

    var hierarchy = AgglomerativeClustering(compute_distances=True)
    _ = hierarchy.fit(features)
    var tree = ward_tree(features, return_distance=True)
    tree.validate()
    var complete = linkage_tree(features)
    complete.validate()
    for i in range(tree.children.size): assert_equal(tree.children.unsafe_get(i), hierarchy.children_.unsafe_get(i))
    var agglomeration = FeatureAgglomeration(1)
    var pooled = agglomeration.fit_transform(features)
    assert_equal(pooled.shape[1], 1)
    assert_equal(agglomeration.inverse_transform(pooled).shape[1], 2)

    var birch = Birch(n_clusters=2)
    _ = birch.partial_fit(features).partial_fit()
    var birch_labels = birch.predict(features)
    for i in range(6): assert_equal(birch_labels.item(i), birch.labels_.value().item(i))
    assert_equal(birch.transform(features).shape[1], birch.subcluster_centers_.shape[0])
