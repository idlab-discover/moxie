"""Cross decomposition inside clone, Pipeline and cross_val_score.

Every composed result is checked exactly against the same steps run by hand,
so composition adds no behavior of its own (sklearn's Pipeline/CV contract).
Scenarios: make_pipeline(StandardScaler(), PLSRegression(...)) predict/score/
feature names/input width in Float32 and Float64 (sklearn users scale before
PLS; plot_pcr_vs_pls fits the same regressor standalone), PLSCanonical and CCA
as pipeline regressors, a cloned pipeline with a typed nested set_params, and
5-fold cross_val_score with the estimator's own R^2 `score`. All estimators
are deterministic, so comparisons are exact. Feature names and the
`input_features` length check follow ClassNamePrefixFeaturesOutMixin.
"""
from numojo import NDArray, Shape, zeros
from std.math import sin, cos
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline, make_pipeline
from moxie.model_selection import IndexArray, IndexSplit, KFold, cross_val_score
from moxie.model_selection._evaluation import TypedScorer
from moxie.model_selection._validation import CVAdapter
from moxie.model_selection._split import _rows

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
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def scaled_pls[dtype: DType](q: Int) raises:
    var x = data[dtype](40, 5)
    var y = targets(x, q)
    var pipe = make_pipeline(StandardScaler[dtype](), PLSRegression[dtype](n_components=2))
    assert_equal(pipe.transformer_name, "standardscaler")
    # sklearn names the step "plsregression"; make_pipeline reflects the
    # `_PLS` struct behind the alias and yields "_pls" (documented gap).
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
    assert_equal(can_pipe.get_feature_names_out()[1], "plscanonical1")
    assert_equal(cca_pipe.get_feature_names_out()[0], "cca0")


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


struct R2(TypedScorer):
    """sklearn's default for a regressor without `scoring`: estimator.score(X, y)."""

    comptime Estimator = PLS
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: PLS, X: Dense, y: Dense, metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


struct Rows(CVAdapter):
    comptime Scorer = R2

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: PLS, X: Dense, y: Dense, metadata: NoneType) raises:
        _ = estimator.fit(X, y)


def cross_validation() raises:
    var x = data[DType.float64](50, 4)
    var y = targets(x, 0)
    var cursor = KFold(5).split(50)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var model = PLS(n_components=2)
    var scores = cross_val_score[Rows](model, x, y, Rows(), R2(), splits, None, None)
    assert_equal(scores.size, 5)
    assert_false(model.__sklearn_is_fitted__())
    for fold in range(5):
        var train = splits[fold].train.copy()
        var test = splits[fold].test.copy()
        var by_hand = PLS(n_components=2)
        _ = by_hand.fit(_rows(x, train), _rows(y, train))
        var expected = by_hand.score(_rows(x, test), _rows(y, test))
        assert_equal(scores.unsafe_get(fold), expected)
        assert_true(expected > 0.5)


def main() raises:
    scaled_pls[DType.float64](0)
    scaled_pls[DType.float64](3)
    scaled_pls[DType.float32](0)
    scaled_pls[DType.float32](2)
    canonical_regressors()
    unfitted_and_plssvd()
    cloned_pipeline()
    cross_validation()
    print("PASS: cross decomposition in Pipeline, clone, nested set_params and cross_val_score")
