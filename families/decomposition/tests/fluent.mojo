"""Shared-trait chaining mutates the original estimator in both dtypes."""
from std.testing import assert_equal, assert_true, assert_false
from numojo import NDArray, Shape, zeros
from moxie.base import UnsupervisedEstimator, TransformerMixin
from moxie.decomposition import (
    PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, NMF, MiniBatchNMF,
    KernelPCA, LatentDirichletAllocation, SparseCoder,
)


def chained[E: UnsupervisedEstimator & TransformerMixin](mut model: E, X: E.Input) raises where conforms_to(E.Target, Deinitable) and conforms_to(E.Transformed, Deinitable):
    var params = model.get_params()
    _ = model.set_params(params).fit(X).transform(X)
    assert_true(model.__sklearn_is_fitted__())


def exercise[dtype: DType]() raises:
    var X = zeros[dtype](Shape(6, 3))
    for i in range(6):
        for j in range(3):
            X.unsafe_set(i * 3 + j, SIMD[dtype, 1](Float64((i * 7 + j * 3) % 5) + Float64(i) / 3))
    var pca = PCA[dtype](2)
    chained(pca, X)
    var ipca = IncrementalPCA[dtype](2, batch_size=3)
    chained(ipca, X)
    _ = ipca.partial_fit(X).transform(X)
    assert_equal(ipca.n_samples_seen_, 12)
    var svd = TruncatedSVD[dtype](2, random_state=0)
    chained(svd, X)
    var fa = FactorAnalysis[dtype](1)
    chained(fa, X)
    var ica = FastICA[dtype](2, random_state=0)
    chained(ica, X)
    var nmf = NMF[dtype](2, random_state=0)
    chained(nmf, X)
    var mini = MiniBatchNMF[dtype](2, random_state=0)
    chained(mini, X)
    var steps = mini.n_steps_
    _ = mini.partial_fit(X).transform(X)
    assert_equal(mini.n_steps_, steps + 1)
    var kpca = KernelPCA[dtype](2, kernel="rbf")
    chained(kpca, X)
    var lda = LatentDirichletAllocation[dtype](2, random_state=0)
    chained(lda, X)
    var batches = lda.n_batch_iter_
    _ = lda.partial_fit(X).transform(X)
    assert_true(lda.n_batch_iter_ > batches)
    var coder = SparseCoder[dtype](X, transform_algorithm="threshold")
    chained(coder, X)
    print("PASS: fluent decomposition", dtype)


def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
