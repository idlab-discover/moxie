"""Build a CF tree incrementally, then cluster its subcenters."""
import numojo as nm
from moxie.cluster import Birch


def main() raises:
    var batch = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        batch[nm.Item(i, 0)] = Float64(0 if i < 3 else 8)
        batch[nm.Item(i, 1)] = Float64(i % 3) * 0.1
    var model = Birch(threshold=0.5, n_clusters=None)
    _ = model.partial_fit(batch)
    _ = model.set_params(n_clusters=Optional[Int](2)).partial_fit()
    print("Subcluster centers:", model.subcluster_centers_)
    print("Predictions:", model.predict(batch))
    print("Distances to subclusters:", model.transform(batch))
    print("Transform names:", model.get_feature_names_out())
