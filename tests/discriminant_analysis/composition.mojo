"""Discriminant analysis inside clone, Pipeline, cross-validation and grid search.

Pipeline results are checked against the steps run by hand. The optional
parity output also compares scaled projections and stratified LDA shrinkage
selection with pinned scikit-learn.
"""
from numojo import NDArray, Shape, zeros
from std.math import sin, cos
from std.sys import argv
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import FitTransformer, clone
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, Shrinkage
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline, TransformingPipeline, make_pipeline
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, IndexArray, cross_val_score, select_rows,
)

comptime LDA = LinearDiscriminantAnalysis[DType.float64, Int]


def blobs[dtype: DType](n: Int, p: Int, offset: Int = 0) raises -> NDArray[dtype]:
    """Three shifted, unevenly scaled classes (label i % 3) with deterministic noise."""
    var x = zeros[dtype](Shape(n, p))
    for row in range(n):
        var i = row + offset
        var c = i % 3
        for j in range(p):
            var noise = sin(Float64(i * 31 + j * 17) * 12.9898) + 0.5 * cos(Float64(i * 7 + j) * 3.1)
            var v = Float64(c) * Float64(j % 2 + 1) * 1.5 + noise * Float64(j + 1) + 10.0 * Float64(j)
            x.unsafe_set(row * p + j, Scalar[dtype](v))
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


def fit_projection[P: FitTransformer](mut pipe: P, X: P.Input, y: Optional[P.Target]) raises -> P.Transformed:
    return pipe.fit_transform(X, y)


def project[P: TransformingPipeline](pipe: P, X: P.Input) raises -> P.Transformed:
    return pipe.transform(X)


def scaled_lda_projection[dtype: DType](emit: Bool) raises:
    var x = blobs[dtype](45, 4)
    var query = blobs[dtype](7, 4, offset=45)
    var y = labels(45)
    var pipe = make_pipeline(
        StandardScaler[dtype](),
        LinearDiscriminantAnalysis[dtype, Int](n_components=2),
    )
    with assert_raises(contains="NotFittedError"):
        _ = pipe.transform(query)
    with assert_raises(contains="requires y"):
        _ = pipe.fit_transform(x)
    assert_false(pipe.transformer.__sklearn_is_fitted__())
    var transformed = fit_projection(pipe, x, Optional(y.copy()))
    var projected = project(pipe, query)
    assert_equal(transformed.shape[0], 45)
    assert_equal(transformed.shape[1], 2)
    assert_equal(projected.shape[0], 7)
    assert_equal(projected.shape[1], 2)
    assert_equal(pipe.n_features_in_(), 4)
    var scaler = StandardScaler[dtype]()
    var scaled = scaler.fit_transform(x)
    var lda = LinearDiscriminantAnalysis[dtype, Int](n_components=2)
    same(transformed, lda.fit_transform(scaled, y.copy()))
    var scaled_query = scaler.transform(query)
    same(projected, lda.transform(scaled_query))
    same_labels(pipe.predict(query), lda.predict(scaled_query))

    var fresh = clone(pipe)
    _ = fresh.fit(x, y)
    same(project(fresh, query), projected)
    if emit:
        for row in range(transformed.shape[0]):
            print("projection", String(dtype), "train", transformed.unsafe_get(2 * row), transformed.unsafe_get(2 * row + 1))
        for row in range(projected.shape[0]):
            print("projection", String(dtype), "query", projected.unsafe_get(2 * row), projected.unsafe_get(2 * row + 1))

    var lsqr = make_pipeline(StandardScaler[dtype](), LinearDiscriminantAnalysis[dtype, Int](solver="lsqr"))
    with assert_raises(contains="NotImplementedError"):
        _ = lsqr.fit_transform(x, y.copy())


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


def score_weights[dtype: DType](n: Int) raises -> NDArray[dtype]:
    var weights = zeros[dtype](Shape(n))
    for i in range(n):
        weights.unsafe_set(i, Scalar[dtype](1 + i % 7))
    return weights^


def routed_scores[dtype: DType](emit: Bool) raises:
    var pipe = make_pipeline(StandardScaler[dtype](), LinearDiscriminantAnalysis[dtype]())
    var x = blobs[dtype](45, 4)
    var y = labels(45)
    var weights = score_weights[dtype](45)
    var standard = {"sample_weight": weights.copy()}
    with assert_raises(contains="NotFittedError"):
        _ = pipe.score(x, y, metadata=standard)

    _ = pipe.fit(x, y)
    # Use the same labels on both reference and native sides, with one flipped.
    y[0] = 1
    var plain = pipe.score(x, y)
    var weighted = pipe.score(x, y, sample_weight=weights.copy())
    assert_true(plain != weighted)
    assert_equal(pipe.score(x, y, metadata=Dict[String, NDArray[dtype]]()), plain)
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = pipe.score(x, y, metadata=standard)

    _ = pipe.estimator.set_score_request(sample_weight=False)
    with assert_raises(contains="unexpected metadata"):
        _ = pipe.score(x, y, metadata=standard)
    _ = pipe.estimator.set_score_request(sample_weight=True)
    assert_equal(pipe.score(x, y, metadata=standard), weighted)

    _ = pipe.estimator.set_score_request(sample_weight=String("validation_weight"))
    var aliased = {"validation_weight": weights.copy()}
    var routed = pipe.score(x, y, metadata=aliased)
    assert_equal(routed, weighted)
    with assert_raises(contains="unexpected metadata"):
        _ = pipe.score(x, y, metadata=standard)

    var qda = make_pipeline(StandardScaler[dtype](), QuadraticDiscriminantAnalysis[dtype]())
    _ = qda.estimator.set_score_request(sample_weight=String("validation_weight"))
    _ = qda.fit(x, labels(45))
    var qda_routed = qda.score(x, y, metadata=aliased)
    assert_equal(qda_routed, qda.score(x, y, sample_weight=weights.copy()))
    assert_true(qda_routed != qda.score(x, y))

    aliased["unexpected"] = weights.copy()
    with assert_raises(contains="unexpected metadata"):
        _ = pipe.score(x, y, metadata=aliased)
    if emit:
        print("routed_score", String(dtype), "lda", routed)
        print("routed_score", String(dtype), "qda", qda_routed)


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


def apply_shrinkage[dtype: DType](
    mut estimator: LinearDiscriminantAnalysis[dtype], params: Dict[String, Float64],
) raises:
    for key in params.keys():
        if key != "shrinkage":
            raise Error("ValueError: unsupported LDA search parameter " + key)
        _ = estimator.set_params(shrinkage=Shrinkage(params[key]))


def cross_validation_and_search[dtype: DType](emit: Bool) raises:
    comptime Model = LinearDiscriminantAnalysis[dtype]
    comptime Adapter = DenseSupervisedAdapter[Model, List[Int], dtype]
    # Grouped, unequal class counts make default stratification observable.
    var order = IndexArray(Shape(61))
    var y = List[Int](capacity=61)
    for label in range(3):
        for source_row in range(label, 61, 3):
            order.unsafe_set(len(y), Int64(source_row))
            y.append(label)
    var x = select_rows(blobs[dtype](61, 4), order)

    var model = Model(solver="lsqr", shrinkage=0.25)
    var adapter = Adapter()
    var scores = cross_val_score[Adapter](model, x, y, adapter)
    var splits = adapter.default_cv(x, y)
    assert_equal(scores.size, 5)
    for fold in range(scores.size):
        ref split = splits[fold]
        var train_x = adapter.input_rows(x, split.train, split.train)
        var train_y = adapter.target_rows(y, split.train)
        var test_x = adapter.input_rows(x, split.test, split.train)
        var test_y = adapter.target_rows(y, split.test)
        var by_hand = clone(model)
        _ = by_hand.fit(train_x, train_y)
        assert_equal(scores.item(fold), by_hand.score(test_x, test_y))
        if emit:
            print("lda_cv", String(dtype), fold, scores.item(fold))
            for row in range(split.test.size):
                print("lda_index", String(dtype), fold, split.test.item(row))

    var grid = Dict[String, List[Float64]]()
    grid["shrinkage"] = [1.0, 0.25, 0.0, 0.7]
    var search = GridSearchCV[Adapter](model, grid, Adapter(apply_shrinkage[dtype]))
    _ = search.fit(x, y)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(search.n_splits_(), 5)
    assert_equal(search.best_index_(), 2)
    assert_equal(search.best_params_()["shrinkage"], 0.0)
    assert_equal(search.best_score_(), 1.0)
    assert_false(clone(search).__sklearn_is_fitted__())

    var expected = Model(solver="lsqr", shrinkage=0.0)
    _ = expected.fit(x, y)
    var prediction = search.predict(x)
    same_labels(prediction, expected.predict(x))
    var results = search.cv_results_()
    for fold in range(scores.size):
        assert_equal(results.scores.test_scores.item(1, fold, 0), scores.item(fold))
    if emit:
        for candidate in range(len(results.params)):
            print("lda_mean", String(dtype), candidate, results.mean_test_score.item(candidate, 0))
            print("lda_rank", String(dtype), candidate, results.rank_test_score.item(candidate, 0))
            for fold in range(scores.size):
                print("lda_fold", String(dtype), candidate, fold, results.scores.test_scores.item(candidate, fold, 0))
        print("lda_best", String(dtype), search.best_index_())
        for row in range(len(prediction)):
            print("lda_predict", String(dtype), prediction[row])


def main() raises:
    comptime assert not conforms_to(
        Pipeline[StandardScaler[], QuadraticDiscriminantAnalysis[]], TransformingPipeline,
    )
    comptime assert not conforms_to(
        Pipeline[StandardScaler[], QuadraticDiscriminantAnalysis[]], FitTransformer,
    )
    scaled_lda[DType.float64]()
    scaled_lda[DType.float32]()
    var args = argv()
    var emit = len(args) == 2 and args[1] == "--composition-parity"
    scaled_lda_projection[DType.float64](emit)
    scaled_lda_projection[DType.float32](emit)
    lda_then_qda[DType.float64]()
    lda_then_qda[DType.float32]()
    routed_scores[DType.float64](emit)
    routed_scores[DType.float32](emit)
    cloned_pipeline()
    cross_validation_and_search[DType.float64](emit)
    cross_validation_and_search[DType.float32](emit)
    print("PASS: discriminant analysis pipeline, clone, cross-validation and grid search")
