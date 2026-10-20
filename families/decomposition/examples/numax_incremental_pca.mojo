"""Streaming dense PCA with an explicitly selected Numax CPU backend."""
from numojo import Shape, zeros
from moxie.decomposition import IncrementalPCA
from moxie_numax import NumaxIncrementalPCABackend


def main() raises:
    var model = IncrementalPCA[DType.float64, NumaxIncrementalPCABackend[4, 3, 2]](
        n_components=2, batch_size=4, whiten=True,
    )
    var batch = zeros[DType.float64](Shape(4, 3))
    for i in range(4):
        batch.unsafe_set(i * 3, Float64(i))
        batch.unsafe_set(i * 3 + 1, Float64(i * i))
        batch.unsafe_set(i * 3 + 2, Float64(i % 2))
    model.partial_fit(batch)
    var last = zeros[DType.float64](Shape(1, 3))
    last.unsafe_set(0, 4)
    last.unsafe_set(1, 16)
    model.partial_fit(last)
    var scores = model.transform(batch)
    print("Samples seen:", model.n_samples_seen_)
    print("Projected shape:", scores.shape[0], scores.shape[1])
