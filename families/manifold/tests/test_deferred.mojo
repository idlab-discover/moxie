"""Exported declarations must reject unsupported operations explicitly."""
from std.testing import assert_raises
from moxie.manifold import (
    MDS, TSNE, SpectralEmbedding, LocallyLinearEmbedding, Isomap,
    locally_linear_embedding, smacof, spectral_embedding,
)


def main() raises:
    with assert_raises(contains="NotImplementedError: MDS.__init__"):
        _ = MDS()
    with assert_raises(contains="NotImplementedError: TSNE.__init__"):
        _ = TSNE()
    with assert_raises(contains="NotImplementedError: SpectralEmbedding.__init__"):
        _ = SpectralEmbedding()
    with assert_raises(contains="NotImplementedError: LocallyLinearEmbedding.__init__"):
        _ = LocallyLinearEmbedding()
    with assert_raises(contains="NotImplementedError: Isomap.__init__"):
        _ = Isomap()
    with assert_raises(contains="NotImplementedError: locally_linear_embedding"):
        _ = locally_linear_embedding(1, n_neighbors=5, n_components=2)
    with assert_raises(contains="NotImplementedError: smacof"):
        _ = smacof(1)
    with assert_raises(contains="NotImplementedError: spectral_embedding"):
        _ = spectral_embedding(1)
    print("PASS: deferred estimator and function declarations raise NotImplementedError")
