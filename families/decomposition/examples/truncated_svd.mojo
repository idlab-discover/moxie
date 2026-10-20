"""Latent semantic analysis of a tiny term-count matrix with TruncatedSVD."""
from numojo import Shape, zeros
from moxie.decomposition import TruncatedSVD


def main() raises:
    # Six documents x five terms; counts are not centered by TruncatedSVD.
    var counts: List[Float64] = [2, 1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 3, 1, 0,
                                 0, 0, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 1]
    var x = zeros[DType.float64](Shape(6, 5))
    for i in range(30): x.unsafe_set(i, counts[i])
    var svd = TruncatedSVD(n_components=2, random_state=42)
    var topics = svd.fit_transform(x)
    print("document-topic shape:", topics.shape[0], topics.shape[1])
    print("explained variance ratio:", svd.explained_variance_ratio_.value())
    print("singular values:", svd.singular_values_.value())
