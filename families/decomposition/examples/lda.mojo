"""Topic model of a tiny document-term count matrix."""
from numojo import Shape, zeros
from moxie.decomposition import LatentDirichletAllocation


def main() raises:
    var x = zeros[DType.float64](Shape(8, 6))
    for d in range(8):
        for w in range(6):
            # Documents 0-3 use words 0-2, documents 4-7 use words 3-5.
            var topical = (d < 4 and w < 3) or (d >= 4 and w >= 3)
            x.unsafe_set(d * 6 + w, Float64((d + w) % 3 + 2) if topical else 0.0)
    var lda = LatentDirichletAllocation(n_components=2, random_state=0)
    var topics = lda.fit_transform(x)
    print("document-topic shape:", topics.shape[0], topics.shape[1])
    print("first document topics:", topics.unsafe_get(0), topics.unsafe_get(1))
    print("perplexity:", lda.perplexity(x))
