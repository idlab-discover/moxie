"""Validate and own externally computed coordinates; no embedding is computed."""
import numojo as nm
from moxie.manifold._plumbing import EmbeddingResult

def main() raises:
    var coordinates = nm.zeros[DType.float64](nm.Shape(3, 2))
    coordinates.unsafe_set(2, 1)
    coordinates.unsafe_set(5, 1)
    var result = EmbeddingResult(coordinates, 3, 2)
    coordinates.unsafe_set(0, 7)
    print("Owned first coordinate:", result.embedding.unsafe_get(0))
    print("Samples:", result.embedding.shape[0])
