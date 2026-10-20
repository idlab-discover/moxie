"""Update a clustering model as successive batches arrive."""
import numojo as nm
from moxie.cluster import MiniBatchKMeans


def main() raises:
    var first = nm.zeros[DType.float64](nm.Shape(4, 1))
    first.unsafe_set(1, 1)
    first.unsafe_set(2, 10)
    first.unsafe_set(3, 11)
    var model = MiniBatchKMeans(2, random_state=42)
    _ = model.partial_fit(first)
    var next_batch = nm.zeros[DType.float64](nm.Shape(2, 1))
    next_batch.unsafe_set(0, 2)
    next_batch.unsafe_set(1, 12)
    _ = model.partial_fit(next_batch)
    print("Centers:", model.cluster_centers_)
    print("Latest batch labels:", model.labels_)
    print("Updates:", model.n_steps_)
    print("Predictions:", model.predict(first))
