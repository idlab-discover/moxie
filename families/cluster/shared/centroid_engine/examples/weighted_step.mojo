"""One weighted centroid step; this is not the KMeans estimator."""
import numojo as nm
from moxie.cluster._centroid_engine import Lloyd

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(2, 1))
    x[nm.Item(1, 0)] = 4
    var centers = nm.zeros[DType.float64](nm.Shape(1, 1))
    var weights = nm.ones[DType.float64](nm.Shape(2))
    weights[nm.Item(1)] = 3
    var engine = Lloyd[DType.float64]()
    var assignment = engine.assign(x, centers, weights)
    var result = engine.update(x, assignment.labels, weights, centers)
    print("Updated center:", result.centers.item(0, 0))
    print("Cluster mass:", result.masses.item(0))
