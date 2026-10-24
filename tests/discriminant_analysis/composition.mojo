"""Discriminant analysis inside clone, Pipeline and cross_val_score.

Every composed result is checked exactly against the same steps run by hand, so
the composition adds no behavior of its own (sklearn's Pipeline/CV contract).
Scenarios: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(...))
(plot_classifier_comparison, plot_nca_dim_reduction), LDA as a supervised
reduction before QDA, a cloned pipeline with a typed nested set_params, and
5-fold cross_val_score with the estimator's own accuracy `score`. All solvers
are deterministic, so comparisons are exact.
"""
from numojo import NDArray, Shape, zeros
from std.math import sin, cos
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline, make_pipeline
from moxie.model_selection import IndexArray, IndexSplit, KFold, cross_val_score
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

comptime Dense = NDArray[DType.float64]
comptime LDA = LinearDiscriminantAnalysis[DType.float64, Int]


def blobs[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    """Three shifted, unevenly scaled classes (label i % 3) with deterministic noise."""
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        var c = i % 3
        for j in range(p):
            var noise = sin(Float64(i * 31 + j * 17) * 12.9898) + 0.5 * cos(Float64(i * 7 + j) * 3.1)
            var v = Float64(c) * Float64(j % 2 + 1) * 1.5 + noise * Float64(j + 1) + 10.0 * Float64(j)
            x.unsafe_set(i * p + j, Scalar[dtype](v))
    return x^


def labels(n: Int) -> List[Int]:
    var y = List[Int](capacity=n)
    for i in range(n):
        y.append(i % 3)
    return y^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def same_labels(a: List[Int], b: List[Int]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def scaled_lda[dtype: DType]() raises:
    # plot_classifier_comparison / plot_nca_dim_reduction: make_pipeline(StandardScaler(), LDA(...)).
    var x = blobs[dtype](45, 4)
    var y = labels(45)
    var pipe = make_pipeline(
        StandardScaler[dtype](),
        LinearDiscriminantAnalysis[dtype, Int](solver="eigen", shrinkage="auto"),
    )
    assert_equal(pipe.transformer_name, "standardscaler")
    assert_equal(pipe.estimator_name, "lineardiscriminantanalysis")
    _ = pipe.fit(x, y)
    var scaler = StandardScaler[dtype]()
    var z = scaler.fit_transform(x)
    var lda = LinearDiscriminantAnalysis[dtype, Int](solver="eigen", shrinkage="auto")
    _ = lda.fit(z, y)
    var zt = scaler.transform(x)
    same_labels(pipe.predict(x), lda.predict(zt))
    same(pipe.predict_proba(x), lda.predict_proba(zt))
    same(pipe.predict_log_proba(x), lda.predict_log_proba(zt))
    same(pipe.decision_function(x), lda.decision_function(zt))
    assert_equal(pipe.score(x, y), lda.score(zt, y))
    # sklearn: n_features_in_ comes from the first step, names from the last step.
    assert_equal(pipe.n_features_in_(), 4)
    var names = pipe.get_feature_names_out()
    assert_equal(len(names), 2)
    assert_equal(names[0], "lineardiscriminantanalysis0")
    assert_equal(names[1], "lineardiscriminantanalysis1")


def lda_then_qda[dtype: DType]() raises:
    # Supervised reduction: LDA(n_components=2) feeds QDA, predicting through transform.
    var x = blobs[dtype](60, 5)
    var y = labels(60)
    var pipe = Pipeline(
        LinearDiscriminantAnalysis[dtype, Int](n_components=2),
        QuadraticDiscriminantAnalysis[dtype, Int](reg_param=0.1),
        transformer_name="lda", estimator_name="qda",
    )
    _ = pipe.fit(x, y)
    var lda = LinearDiscriminantAnalysis[dtype, Int](n_components=2)
    var z = lda.fit_transform(x, y.copy())
    var qda = QuadraticDiscriminantAnalysis[dtype, Int](reg_param=0.1)
    _ = qda.fit(z, y)
    var zt = lda.transform(x)
    same_labels(pipe.predict(x), qda.predict(zt))
    same(pipe.predict_proba(x), qda.predict_proba(zt))
    same(pipe.decision_function(x), qda.decision_function(zt))
    assert_equal(pipe.n_features_in_(), 5)
    assert_equal(pipe.transformer.get_n_features_in(), 5)
    assert_equal(pipe.estimator.get_n_features_in(), 2)


def cloned_pipeline() raises:
    var x = blobs[DType.float64](45, 4)
    var y = labels(45)
    var pipe = make_pipeline(StandardScaler(), LDA(solver="svd"))
    _ = pipe.fit(x, y)
    var fresh = clone(pipe)
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = fresh.estimator.get_n_features_in()
    # Typed nested set_params (sklearn: set_params(lineardiscriminantanalysis__solver="lsqr")).
    var params = fresh.get_params()
    params.estimator.solver = "lsqr"
    params.estimator.shrinkage = 0.3
    _ = fresh.set_params(params)
    _ = fresh.fit(x, y)
    var scaler = StandardScaler()
    var z = scaler.fit_transform(x)
    var lda = LDA(solver="lsqr", shrinkage=0.3)
    _ = lda.fit(z, y)
    same_labels(fresh.predict(x), lda.predict(scaler.transform(x)))
    assert_equal(pipe.estimator.solver, "svd")
    assert_true(pipe.estimator.xbar_.__bool__())
    assert_false(fresh.estimator.xbar_.__bool__())


struct Accuracy(TypedScorer):
    """sklearn's default for a classifier without `scoring`: estimator.score(X, y)."""

    comptime Estimator = LDA
    comptime Input = Dense
    comptime Target = List[Int]
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: LDA, X: Dense, y: List[Int], metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct Rows(CVAdapter):
    comptime Scorer = Accuracy

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: List[Int]) raises -> Int:
        return len(y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: List[Int], rows: IndexArray) raises -> List[Int]:
        var out = List[Int](capacity=rows.size)
        for i in range(rows.size):
            out.append(y[Int(rows.unsafe_get(i))])
        return out^

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: LDA, X: Dense, y: List[Int], metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def cross_validation() raises:
    var x = blobs[DType.float64](60, 4)
    var y = labels(60)
    var cursor = KFold(5).split(60)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = LDA(solver="lsqr", shrinkage="auto")
    var scores = cross_val_score[Rows](model, x, y, Rows(), Accuracy(), splits, None, None)
    assert_equal(scores.size, 5)
    assert_false(model.__sklearn_is_fitted__())
    for fold in range(5):
        var train = splits[fold].train.copy()
        var test = splits[fold].test.copy()
        var by_hand = LDA(solver="lsqr", shrinkage="auto")
        _ = by_hand.fit(_rows(x, train), Rows().target_rows(y, train))
        assert_equal(scores.unsafe_get(fold), by_hand.score(_rows(x, test), Rows().target_rows(y, test)))


def main() raises:
    scaled_lda[DType.float64]()
    scaled_lda[DType.float32]()
    lda_then_qda[DType.float64]()
    lda_then_qda[DType.float32]()
    cloned_pipeline()
    cross_validation()
    print("PASS: discriminant analysis composition (pipeline, clone, cross_val_score)")
