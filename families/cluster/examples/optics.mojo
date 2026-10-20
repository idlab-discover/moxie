"""Fit once, inspect density order, and extract another distance threshold."""
import numojo as nm
from moxie.cluster import OPTICS, cluster_optics_dbscan
from moxie.cluster._optics import OPTICSSize


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(8, 1))
    var positions = [1, 2, 3, 1, 8, 8, 7, 100]
    for i in range(8): x[nm.Item(i, 0)] = Float64(positions[i])
    var model = OPTICS(min_samples=OPTICSSize(3))
    var labels = model.fit_predict(x)
    print("Xi labels:", labels)
    print("Density ordering:", model.ordering_)
    print("Reachability in sample order:", model.reachability_)
    var cut = cluster_optics_dbscan(reachability=model.reachability_,
        core_distances=model.core_distances_, ordering=model.ordering_, eps=2)
    print("Labels at distance 2:", cut)
