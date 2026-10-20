"""Discover groups and predict a cluster for a new sample."""
import numojo as nm
from moxie.cluster import MeanShift

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6,2))
    var values: List[Float64] = [1,1,2,1,1,0,4,7,3,5,3,6]
    for i in range(12): x.unsafe_set(i,values[i])
    var model = MeanShift(bandwidth=2.0)
    var labels = model.fit_predict(x)
    print("Centers:", model.cluster_centers_)
    print("Labels:", labels)
    var new_sample = nm.zeros[DType.float64](nm.Shape(1,2))
    print("Cluster for [0,0]:", model.predict(new_sample))
