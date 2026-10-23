"""Decomposition estimators inside clone, Pipeline, FeatureUnion, cross_val_score and GridSearchCV.

Every composed result is checked exactly against the same steps run by hand, so
the composition adds no behavior of its own (sklearn's Pipeline/union/CV/search
contract). The model-selection part mirrors plot_pca_vs_fa_model_selection.py:
`cross_val_score(PCA(n_components=n), X)` and the FactorAnalysis equivalent use
the estimators' own log-likelihood `score`, five unshuffled folds, and
`GridSearchCV(PCA(), {"n_components": ...})` refits the best rank on all rows.
All factorizations here are deterministic (full SVD / lapack-free Jacobi), so
comparisons are exact.
"""
from numojo import NDArray, Shape, zeros
from std.math import sin, cos
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.decomposition import PCA, TruncatedSVD, FactorAnalysis
from moxie.decomposition._pca import ComponentCount
from moxie.linear_model import Ridge, RidgeClassifier
from moxie.pipeline import Pipeline, make_pipeline, make_union
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter, SearchScorer,
    cross_val_score,
)
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]


def low_rank[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    """Rank-3 signal plus deterministic unit-scale noise (the gallery's homoscedastic case)."""
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        var a = sin(Float64(i) * 0.7)
        var b = cos(Float64(i) * 1.3)
        var c = sin(Float64(i * i % 17) * 0.4)
        for j in range(p):
            var noise = sin(Float64(i * 31 + j * 17) * 12.9898) * 0.3
            var v = 3 * a * Float64(j % 3 + 1) + 2 * b * Float64((j + 1) % 4) - c * Float64(j % 2) + noise + 0.5 * Float64(j)
            x.unsafe_set(i * p + j, Scalar[dtype](v))
    return x^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def kfold(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(5).split(n)
    var out = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


def clone_and_refit() raises:
    var x = low_rank[DType.float64](30, 6)
    var model = PCA(2, svd_solver="full")
    _ = model.fit(x)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = fresh.transform(x)
    assert_equal(fresh.get_params().svd_solver, "full")
    _ = fresh.fit(x)
    same(fresh.components_.value(), model.components_.value())
    # set_params then refit replaces the learned rank; the original stays intact.
    _ = fresh.set_params(n_components=ComponentCount(3))
    _ = fresh.fit(x)
    assert_equal(fresh.components_.value().shape[0], 3)
    assert_equal(model.components_.value().shape[0], 2)


def pipelines[dtype: DType]() raises:
    # plot_pcr_vs_pls: principal component regression; plot_digits_pipe: PCA then a classifier.
    var x = low_rank[dtype](40, 6)
    var y = zeros[dtype](Shape(40))
    var labels = zeros[DType.int64](Shape(40))
    for i in range(40):
        y.unsafe_set(i, Scalar[dtype](sin(Float64(i) * 0.7) * 2 + Float64(i % 3)))
        labels.unsafe_set(i, Int64(i % 2))
    var pcr = make_pipeline(PCA[dtype](2), Ridge[dtype](alpha=0.5))
    assert_equal(pcr.transformer_name, "pca")
    assert_equal(pcr.estimator_name, "ridge")
    _ = pcr.fit(x, y)
    var pca = PCA[dtype](2)
    var z = pca.fit_transform(x)
    var ridge = Ridge[dtype](alpha=0.5)
    _ = ridge.fit(z, y)
    # sklearn fits through fit_transform but predicts through transform.
    var projected = pca.transform(x)
    same(pcr.predict(x), ridge.predict(projected))
    assert_equal(pcr.score(x, y), ridge.score(projected, y))
    assert_equal(pcr.n_features_in_(), 6)
    var names = pcr.transformer.get_feature_names_out()
    assert_equal(names[1], "pca1")

    var clf = Pipeline(TruncatedSVD[dtype](3, algorithm="arpack"), RidgeClassifier[dtype](),
                       transformer_name="svd", estimator_name="clf")
    _ = clf.fit(x, labels)
    var svd = TruncatedSVD[dtype](3, algorithm="arpack")
    var features = svd.fit_transform(x)
    var classifier = RidgeClassifier[dtype]()
    _ = classifier.fit(features, labels)
    same(clf.predict(x), classifier.predict(svd.transform(x)))
    # Clone: configuration copied, nothing fitted; typed nested set_params then refit.
    var fresh = clone(clf)
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.n_features_in_()
    var params = fresh.get_params()
    params.transformer.n_components = 2
    _ = fresh.set_params(params)
    _ = fresh.fit(x, labels)
    assert_equal(fresh.transformer.components_.value().shape[0], 2)
    assert_equal(clf.transformer.components_.value().shape[0], 3)


def union[dtype: DType]() raises:
    # plot_feature_union: FeatureUnion([("pca", PCA(...)), ...]) stacks the branch outputs.
    var x = low_rank[dtype](25, 5)
    var u = make_union[dtype=dtype](PCA[dtype](2), TruncatedSVD[dtype](2, algorithm="arpack"))
    var stacked = u.fit_transform(x)
    assert_equal(stacked.shape[0], 25)
    assert_equal(stacked.shape[1], 4)
    var pca = PCA[dtype](2)
    var a = pca.fit_transform(x)
    var svd = TruncatedSVD[dtype](2, algorithm="arpack")
    var b = svd.fit_transform(x)
    for i in range(25):
        for j in range(2):
            assert_equal(stacked.unsafe_get(i * 4 + j), a.unsafe_get(i * 2 + j))
            assert_equal(stacked.unsafe_get(i * 4 + 2 + j), b.unsafe_get(i * 2 + j))
    var names = u.get_feature_names_out()
    assert_equal(len(names), 4)
    assert_equal(names[0], "pca__pca0")
    assert_equal(names[3], "truncatedsvd__truncatedsvd1")
    assert_equal(u.n_features_in_(), 5)


struct PCALogLikelihood(SearchScorer):
    """sklearn's default scorer for an unsupervised estimator: estimator.score(X)."""
    comptime Estimator = PCA[]
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: PCA[], X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X)


struct PCASearch(SearchAdapter):
    """Tunes n_components, as GridSearchCV(PCA(), {"n_components": n_components})."""
    comptime Scorer = PCALogLikelihood
    comptime Value = Int

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return X.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: PCA[], X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X)

    def configure(self, mut estimator: PCA[], params: Dict[String, Int]) raises:
        for key in params.keys():
            if key != "n_components":
                raise Error("ValueError: Invalid parameter " + key + " for estimator PCA")
            _ = estimator.set_params(n_components=ComponentCount(params[key]))

    def default_scorer(self) raises -> PCALogLikelihood:
        return PCALogLikelihood()

    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return kfold(X.shape[0])


struct FALogLikelihood(SearchScorer):
    comptime Estimator = FactorAnalysis[]
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def score(self, estimator: FactorAnalysis[], X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X)


struct FACV(SearchAdapter):
    comptime Scorer = FALogLikelihood
    comptime Value = Int

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return X.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: FactorAnalysis[], X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X)

    def configure(self, mut estimator: FactorAnalysis[], params: Dict[String, Int]) raises:
        for key in params.keys():
            if key != "n_components":
                raise Error("ValueError: Invalid parameter " + key + " for estimator FactorAnalysis")
            _ = estimator.set_params(n_components=Optional[Int](params[key]))

    def default_scorer(self) raises -> FALogLikelihood:
        return FALogLikelihood()

    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return kfold(X.shape[0])


def model_selection() raises:
    var x = low_rank[DType.float64](60, 8)
    var y = zeros[DType.float64](Shape(60))  # ignored, as sklearn's y=None
    var splits = kfold(60)
    var ranks: List[Int] = [1, 2, 3, 4, 5]
    var pca_means = List[Float64]()
    for r in ranks:
        var template = PCA(r, svd_solver="full")
        var scores = cross_val_score[PCASearch](template, x, y, PCASearch(), PCALogLikelihood(), splits, None, None)
        assert_equal(scores.size, 5)
        var total: Float64 = 0
        for k in range(5):
            var fold = clone(template)
            _ = fold.fit(_rows(x, splits[k].train))
            assert_equal(scores.item(k), fold.score(_rows(x, splits[k].test)))
            total += scores.item(k)
        pca_means.append(total / 5)
        var fa_template = FactorAnalysis(r)
        var fa_scores = cross_val_score[FACV](fa_template, x, y, FACV(), FALogLikelihood(), splits, None, None)
        for k in range(5):
            var fold = clone(fa_template)
            _ = fold.fit(_rows(x, splits[k].train))
            assert_equal(fa_scores.item(k), fold.score(_rows(x, splits[k].test)))

    var grid = Dict[String, List[Int]]()
    grid["n_components"] = ranks.copy()
    var search = GridSearchCV[PCASearch](PCA(svd_solver="full"), grid, PCASearch())
    search.fit(x, y, None, None)
    var results = search.cv_results_().copy()
    assert_equal(len(results.params), 5)
    var best = 0
    for i in range(5):
        assert_equal(results.params[i]["n_components"], ranks[i])
        assert_equal(results.mean_test_score.item(i), pca_means[i])
        if pca_means[i] > pca_means[best]:
            best = i
    assert_equal(search.best_params_()["n_components"], ranks[best])
    var refit = PCA(ranks[best], svd_solver="full")
    _ = refit.fit(x)
    same(search.best_estimator_().components_.value(), refit.components_.value())


def main() raises:
    clone_and_refit()
    pipelines[DType.float32]()
    pipelines[DType.float64]()
    union[DType.float32]()
    union[DType.float64]()
    model_selection()
    print("PASS: decomposition clone, Pipeline, FeatureUnion, cross_val_score and GridSearchCV composition")
