"""Density clustering with duplicates and inclusive radius boundaries."""
import numojo as nm
from moxie.cluster import DBSCAN


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    x[nm.Item(1, 0)] = 1
    x[nm.Item(2, 0)] = 1
    x[nm.Item(3, 0)] = 10
    var model = DBSCAN(1, min_samples=2)
    _ = model.fit(x)
    for i in range(4): print("sample", i, "cluster", model.labels_.item(i))
    print("core samples", model.core_sample_indices_.size)
