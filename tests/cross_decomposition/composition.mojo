"""Cross decomposition inside clone, Pipeline, cross_val_score and GridSearchCV.

Every composed result is checked exactly against the same steps run by hand,
so composition adds no behavior of its own (sklearn's Pipeline/CV contract).
Scenarios: make_pipeline(StandardScaler(), PLSRegression(...)) predict/score/
feature names/input width in Float32 and Float64 (sklearn users scale before
PLS; plot_pcr_vs_pls fits the same regressor standalone), PLSCanonical and CCA
as pipeline regressors, a cloned pipeline with a typed nested set_params, and
default 5-fold cross_val_score and GridSearchCV with the estimator's own R^2
`score`. Each fold is checked exactly against a native manual fit. Feature names and the
`input_features` length check follow ClassNamePrefixFeaturesOutMixin.
"""
from numojo import Item, NDArray, Shape, zeros
from std.math import abs, cos, sin
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    DenseSupervisedAdapter,
    ErrorPolicy,
    GridSearchCV,
    KFold,
    cross_val_score,
    select_rows,
)

comptime Dense = NDArray[DType.float64]
comptime PLS = PLSRegression[DType.float64]


def data[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    """Unevenly scaled, correlated columns with deterministic noise."""
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        var latent = sin(Float64(i) * 0.37) * 2.0
        for j in range(p):
            var noise = sin(Float64(i * 31 + j * 17) * 12.9898) + 0.5 * cos(Float64(i * 7 + j) * 3.1)
            var v = latent * Float64(j % 3 + 1) + noise * Float64(j + 1) + 5.0 * Float64(j)
            x.unsafe_set(i * p + j, Scalar[dtype](v))
    return x^


def targets[dtype: DType](x: NDArray[dtype], q: Int) raises -> NDArray[dtype]:
    """(n,) when q == 0, else (n, q): linear in X plus deterministic noise."""
    var n = x.shape[0]
    var p = x.shape[1]
    var y = zeros[dtype](Shape(n)) if q == 0 else zeros[dtype](Shape(n, q))
    var cols = 1 if q == 0 else q
    for i in range(n):
        for t in range(cols):
            var v = 3.0 + Float64(t)
            for j in range(p):
                v += Float64(x.unsafe_get(i * p + j)) * Float64((j + t) % 4 - 1) * 0.5
            v += cos(Float64(i * 13 + t) * 1.7)
            y.unsafe_set(i * cols + t, Scalar[dtype](v))
    return y^


def same(a: Dense, b: Dense) raises:
    assert_equal(a.ndim, b.ndim)
    for axis in range(a.ndim):
        assert_equal(a.shape[axis], b.shape[axis])
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def scaled_pls[dtype: DType](q: Int) raises:
    var x = data[dtype](40, 5)
    var y = targets(x, q)
    var pipe = make_pipeline(StandardScaler[dtype](), PLSRegression[dtype](n_components=2))
    assert_equal(pipe.transformer_name, "standardscaler")
    assert_equal(pipe.estimator_name, "plsregression")
    _ = pipe.fit(x, y)
    var scaler = StandardScaler[dtype]()
    var z = scaler.fit_transform(x)
    var pls = PLSRegression[dtype](n_components=2)
    _ = pls.fit(z, y)
    var zt = scaler.transform(x)
    same(pipe.predict(x), pls.predict(zt))
    assert_equal(pipe.predict(x).ndim, 1 if q == 0 else 2)
    assert_equal(pipe.score(x, y), pls.score(zt, y))
    # sklearn: names come from the last step, the width from the first.
    var names = pipe.get_feature_names_out()
    assert_equal(len(names), 2)
    assert_equal(names[0], "plsregression0")
    assert_equal(names[1], "plsregression1")
    assert_equal(pls.get_n_features_in(), 5)
    var given = List[String]()
    for j in range(5):
        given.append("x" + String(j))
    assert_equal(pls.get_feature_names_out(given^)[1], "plsregression1")
    with assert_raises(contains="input_features should have length equal to number of features (5), got 1"):
        _ = pls.get_feature_names_out(List[String](["x0"]))


def canonical_regressors() raises:
    # PLSCanonical and CCA are RegressorMixin in sklearn too.
    var x = data[DType.float64](40, 4)
    var y = targets(x, 2)
    var can_pipe = make_pipeline(StandardScaler(), PLSCanonical(2, algorithm="svd"))
    var cca_pipe = make_pipeline(StandardScaler(), CCA(1))
    _ = can_pipe.fit(x, y)
    _ = cca_pipe.fit(x, y)
    var scaler = StandardScaler()
    var z = scaler.fit_transform(x)
    var can = PLSCanonical(2, algorithm="svd")
    var cca = CCA(1)
    _ = can.fit(z, y)
    _ = cca.fit(z, y)
    same(can_pipe.predict(x), can.predict(z))
    same(cca_pipe.predict(x), cca.predict(z))
    assert_equal(can_pipe.estimator_name, "plscanonical")
    assert_equal(cca_pipe.estimator_name, "cca")
    assert_equal(can_pipe.get_feature_names_out()[1], "plscanonical1")
    assert_equal(cca_pipe.get_feature_names_out()[0], "cca0")

    _ = cca_pipe.estimator.set_params(max_iter=1)
    _ = cca_pipe.fit(x, y)
    assert_equal(len(cca_pipe.estimator.fit_warnings), 1)
    assert_equal(cca_pipe.estimator.fit_warnings[0].category, "ConvergenceWarning")
    assert_equal(
        cca_pipe.estimator.fit_warnings[0].message,
        "Maximum number of iterations reached",
    )


def aliased_pipeline_names() raises:
    var distinct = make_pipeline(PLSRegression(1), CCA(1))
    assert_equal(distinct.transformer_name, "plsregression")
    assert_equal(distinct.estimator_name, "cca")
    var repeated = make_pipeline(PLSRegression(1), PLSRegression(1))
    assert_equal(repeated.transformer_name, "plsregression-1")
    assert_equal(repeated.estimator_name, "plsregression-2")
    var svd = make_pipeline(PLSSVD(1), PLSRegression(1))
    assert_equal(svd.transformer_name, "plssvd")


def unfitted_and_plssvd() raises:
    var x = data[DType.float64](20, 3)
    var y = targets(x, 2)
    var reg = PLS(1)
    with assert_raises(contains="NotFittedError"):
        _ = reg.get_n_features_in()
    var svd = PLSSVD(2)
    with assert_raises(contains="NotFittedError"):
        _ = svd.get_n_features_in()
    with assert_raises(contains="NotFittedError"):
        _ = svd.get_feature_names_out()
    _ = svd.fit(x, y)
    assert_equal(svd.get_n_features_in(), 3)
    var names = svd.get_feature_names_out(List[String](["a", "b", "c"]))
    assert_equal(len(names), 2)
    assert_equal(names[1], "plssvd1")
    with assert_raises(contains="input_features should have length equal to number of features (3), got 2"):
        _ = svd.get_feature_names_out(List[String](["a", "b"]))


def cloned_pipeline() raises:
    var x = data[DType.float64](40, 5)
    var y = targets(x, 2)
    var pipe = make_pipeline(StandardScaler(), PLS(n_components=1))
    _ = pipe.fit(x, y)
    var fresh = clone(pipe)
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict(x)
    # Typed nested set_params (sklearn: set_params(plsregression__n_components=3, plsregression__scale=False)).
    var params = fresh.get_params()
    params.estimator.n_components = 3
    params.estimator.scale = False
    _ = fresh.set_params(params)
    _ = fresh.fit(x, y)
    var scaler = StandardScaler()
    var z = scaler.fit_transform(x)
    var pls = PLS(n_components=3, scale=False)
    _ = pls.fit(z, y)
    same(fresh.predict(x), pls.predict(scaler.transform(x)))
    assert_equal(fresh.estimator.x_weights_.shape[1], 3)
    assert_equal(pipe.estimator.n_components, 1)
    assert_equal(pipe.estimator.x_weights_.shape[1], 1)


def configure_components[dtype: DType](
    mut estimator: PLSRegression[dtype], params: Dict[String, Int]
) raises:
    for name in params.keys():
        if name != "n_components":
            raise Error("Unexpected PLS parameter: " + name)
        _ = estimator.set_params(n_components=params[name])


def cross_validation_and_search[dtype: DType](q: Int) raises:
    comptime Regression = PLSRegression[dtype]
    comptime Adapter = DenseSupervisedAdapter[
        Regression, NDArray[dtype], dtype, Int, weight_dtype=DType.float64
    ]
    var x = data[dtype](53, 4)
    var y = targets(x, q)
    var model = Regression(n_components=2)
    var error_policy = ErrorPolicy(True, 0)
    var scores = cross_val_score[Adapter](model, x, y, Adapter(), error_score=error_policy)

    var candidates: List[Int] = [3, 1, 2]
    var grid = Dict[String, List[Int]]()
    grid["n_components"] = candidates.copy()
    var search = GridSearchCV[Adapter](
        model, grid, Adapter(configure_components[dtype]), error_score=error_policy
    )
    _ = search.fit(x, y)
    var results = search.cv_results_()
    assert_equal(scores.size, 5)
    assert_equal(search.n_splits_(), 5)
    assert_equal(len(results.params), len(candidates))
    assert_equal(results.scores.test_scores.shape[2], 1)

    var best_index = 0
    var best_mean = -Float64.MAX
    for candidate in range(len(candidates)):
        var count = candidates[candidate]
        assert_equal(results.params[candidate]["n_components"], count)
        var cursor = KFold(5).split(x.shape[0])
        var total: Float64 = 0
        for fold in range(5):
            var next_split = cursor.next()
            var split = next_split.take()
            var by_hand = Regression(n_components=count)
            _ = by_hand.fit(select_rows(x, split.train), select_rows(y, split.train))
            var expected = by_hand.score(
                select_rows(x, split.test), select_rows(y, split.test)
            )
            var actual = results.scores.test_scores[Item(candidate, fold, 0)]
            assert_equal(actual, expected)
            if count == model.n_components:
                assert_equal(scores[Item(fold)], expected)
            total += expected
            print("fold", q, count, fold, actual)

        var mean = total / 5
        assert_true(abs(results.mean_test_score[Item(candidate, 0)] - mean) < 1e-14)
        if mean > best_mean:
            best_index = candidate
            best_mean = mean

    assert_equal(search.best_index_(), best_index)
    assert_equal(search.best_params_()["n_components"], candidates[best_index])
    assert_true(abs(search.best_score_() - best_mean) < 1e-14)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(model.n_components, 2)
    assert_false(clone(search).__sklearn_is_fitted__())
    print("winner", q, search.best_index_(), search.best_score_())

    var refit = Regression(n_components=candidates[best_index])
    _ = refit.fit(x, y)
    var prediction = search.predict(x)
    same(prediction, refit.predict(x))
    for i in range(prediction.size):
        print("predict", q, i, prediction.unsafe_get(i))


def main() raises:
    scaled_pls[DType.float64](0)
    scaled_pls[DType.float64](3)
    scaled_pls[DType.float32](0)
    scaled_pls[DType.float32](2)
    canonical_regressors()
    aliased_pipeline_names()
    unfitted_and_plssvd()
    cloned_pipeline()
    cross_validation_and_search[DType.float64](0)
    cross_validation_and_search[DType.float32](2)
    print("PASS: cross decomposition in Pipeline, clone, cross_val_score and GridSearchCV")
