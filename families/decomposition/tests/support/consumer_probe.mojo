"""Isolated-consumer probe: fit each implemented estimator on one tiny matrix."""
from numojo import Shape, zeros
from moxie.decomposition import PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, NMF, SparseCoder, KernelPCA, LatentDirichletAllocation, randomized_svd


def main() raises:
    var x = zeros[DType.float64](Shape(6, 3))
    for i in range(6):
        for j in range(3): x.unsafe_set(i * 3 + j, Float64((i * 7 + j * 3) % 5) + Float64(i) / 3)
    var pca = PCA(n_components=2)
    pca.fit(x)
    var ipca = IncrementalPCA(n_components=2, batch_size=3)
    ipca.fit(x)
    var factors = randomized_svd(x, 2, random_state=0)
    var tsvd = TruncatedSVD(2, random_state=0)
    tsvd.fit(x)
    var fa = FactorAnalysis(1)
    fa.fit(x)
    var ica = FastICA(2, random_state=0)
    ica.fit(x)
    var nmf = NMF(2, random_state=0)
    nmf.fit(x)
    var coder = SparseCoder(x, transform_algorithm="threshold")
    assert coder.transform(x).shape[1] == 6
    var kpca = KernelPCA(2, kernel="rbf")
    assert kpca.fit_transform(x).shape[1] == 2
    var lda = LatentDirichletAllocation(2, random_state=0)
    lda.fit(x)
    assert lda.transform(x).shape[1] == 2
    print("probe", nmf.n_iter_ > 0, fa.n_iter_ > 0, ica.n_iter_ > 0, pca.n_components_, ipca.n_components_, factors.singular_values.size, tsvd.components_.value().shape[0])
