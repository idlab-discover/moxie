"""Exported declarations must reject unsupported operations explicitly."""
from std.testing import assert_raises
from moxie.manifold import MDS, TSNE, SpectralEmbedding, LocallyLinearEmbedding, Isomap


def main() raises:
    with assert_raises():
        _ = MDS()
    with assert_raises():
        _ = TSNE()
    with assert_raises():
        _ = SpectralEmbedding()
    with assert_raises():
        _ = LocallyLinearEmbedding()
    with assert_raises():
        _ = Isomap()
    print("PASS: deferred estimator declarations reject construction")
