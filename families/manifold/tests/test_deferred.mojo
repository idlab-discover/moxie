"""Exported declarations must reject unsupported operations explicitly."""
from std.testing import assert_raises
from moxie.manifold import SpectralEmbedding, spectral_embedding


def main() raises:
    with assert_raises(contains="NotImplementedError: SpectralEmbedding.__init__"):
        _ = SpectralEmbedding()
    with assert_raises(contains="NotImplementedError: spectral_embedding"):
        _ = spectral_embedding(1)
    print("PASS: deferred estimator and function declarations raise NotImplementedError")
