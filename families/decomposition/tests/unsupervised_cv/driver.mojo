"""Exercise the targetless CV overloads with real unsupervised estimators."""
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from moxie.decomposition import FactorAnalysis, LatentDirichletAllocation, PCA
from moxie.model_selection import (
    DenseUnsupervisedAdapter, ErrorPolicy, IndexSplit, cross_val_score, cross_validate,
)
from moxie.neighbors import KernelDensity
from numojo import NDArray, Shape, zeros
from decomposition_testing import emit, read_matrix


def evaluate[dtype: DType](X: NDArray[dtype]) raises:
    var model = PCA[dtype](2, svd_solver="full")
    var default_scores = cross_val_score(model, X)
    assert_equal(default_scores.shape, Shape(5))
    emit("pca_default", default_scores)
    emit("pca_three", cross_val_score(model, X, cv=3))

    comptime Adapter = DenseUnsupervisedAdapter[PCA[dtype], dtype]
    var adapter = Adapter(4)
    var folds = adapter.default_cv(X, None)
    var reordered = List[IndexSplit]()
    reordered.append(folds[3].copy())
    reordered.append(folds[0].copy())
    _ = model.fit(X)
    var previous_score = model.score(X)
    var scores = cross_val_score(model, X, cv=reordered)
    assert_equal(scores.shape, Shape(2))
    emit("pca_explicit", scores)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.score(X), previous_score)

    var detailed = cross_validate[Adapter](
        model, X, None, adapter, fit_metadata=None, score_metadata=None,
        cv=reordered^, return_estimator=True, return_indices=True,
    )
    assert_equal(detailed.estimators.value()[0].n_samples_, X.shape[0] - folds[3].test.size)
    assert_equal(detailed.indices.value()[0].test.item(0), folds[3].test.item(0))

    emit("fa", cross_val_score(FactorAnalysis[dtype](2, svd_method="lapack"), X))
    emit("lda", cross_val_score(
        LatentDirichletAllocation[dtype](n_components=2, random_state=0, max_iter=3), X, cv=3,
    ))
    emit("kde", cross_val_score(KernelDensity[dtype](bandwidth=1.3), X))

    with assert_raises(contains="rank-two"):
        _ = cross_val_score(model, zeros[dtype](Shape(9)))
    with assert_raises(contains="at least one split"):
        _ = cross_val_score(model, X, cv=List[IndexSplit]())
    with assert_raises(contains="all cross-validation fits failed"):
        _ = cross_val_score(PCA[dtype](X.shape[1] + 1, svd_solver="full"), X)
    with assert_raises(contains="n_components"):
        _ = cross_val_score(
            PCA[dtype](X.shape[1] + 1, svd_solver="full"), X, error_score=ErrorPolicy(True, 0),
        )


def main() raises:
    var args = argv()
    if args[1] == "f32":
        evaluate(read_matrix[DType.float32](String(args[2])))
    else:
        evaluate(read_matrix[DType.float64](String(args[2])))
