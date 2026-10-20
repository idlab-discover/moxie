"""Fit three groups and route unseen rows through the learned binary tree."""
import numojo as nm
from moxie.cluster import BisectingKMeans


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(9, 2))
    for i in range(9):
        x[nm.Item(i, 0)] = Float64((i // 3) * 10)
        x[nm.Item(i, 1)] = Float64(i % 3)
    var model = BisectingKMeans(3, random_state=42,
                               bisecting_strategy="largest_cluster")
    _ = model.fit(x)
    var query = nm.zeros[DType.float64](nm.Shape(2, 2))
    query[nm.Item(0, 0)] = 1
    query[nm.Item(1, 0)] = 19
    print("Training labels:", model.labels_)
    print("Centers:", model.cluster_centers_)
    print("Tree predictions:", model.predict(query))
    print("Distances to final centers:", model.transform(query))
