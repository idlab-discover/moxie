"""Weighted initialization and dense clustering with the same seed."""
import numojo as nm
from moxie.cluster import KMeans, kmeans_plusplus


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        x[nm.Item(i, 0)] = Float64(10 * (i // 3))
        x[nm.Item(i, 1)] = Float64(i % 3)
    var weights = nm.ones[DType.float64](nm.Shape(6))
    weights[nm.Item(0)] = 2
    var seeds = kmeans_plusplus(x, 2, sample_weight=weights.view(), random_state=42)
    print("Initial row indices:", seeds.indices)
    var model = KMeans(2, n_init=4, algorithm="elkan", random_state=42)
    _ = model.fit(x, sample_weight=weights.view())
    print("Centers:", model.cluster_centers_)
    print("Labels:", model.labels_)
    print("Weighted inertia:", model.inertia_)
    print("Score:", model.score(x, sample_weight=weights.view()))
    print("Distances:", model.transform(x))
