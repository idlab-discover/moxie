"""BernoulliRBM inside clone, Pipeline and FeatureUnion.

plot_rbm_logistic_classification.py builds Pipeline([("rbm", BernoulliRBM),
("logistic", LogisticRegression)]). LogisticRegression is not implemented
(#8 part 1), so RidgeClassifier stands in as the supervised final step. Every
composed result is checked exactly against the same steps run by hand: sklearn's
Pipeline fits the transformer through fit_transform and predicts through
transform, and a union stacks branch outputs with prefixed names.
"""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.decomposition import PCA
from moxie.linear_model import RidgeClassifier
from moxie.neural_network import BernoulliRBM
from moxie.pipeline import Pipeline, make_pipeline, make_union


def binary[dtype: DType](n: Int, d: Int) raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(n, d))
    for i in range(n):
        for j in range(d):
            X.unsafe_set(i * d + j, SIMD[dtype, 1](Int((i * 5 + j * 3 + i * j) % 7 < 3)))
    return X^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def pipelines[dtype: DType]() raises:
    var X = binary[dtype](30, 8)
    var labels = zeros[DType.int64](Shape(30))
    for i in range(30):
        labels.unsafe_set(i, Int64(i % 3))
    var pipe = make_pipeline(BernoulliRBM[dtype](6, learning_rate=0.06, n_iter=5, random_state=0),
                             RidgeClassifier[dtype]())
    assert_equal(pipe.transformer_name, "bernoullirbm")
    assert_equal(pipe.estimator_name, "ridgeclassifier")
    _ = pipe.fit(X, labels)
    var rbm = BernoulliRBM[dtype](6, learning_rate=0.06, n_iter=5, random_state=0)
    var features = rbm.fit_transform(X)
    var ridge = RidgeClassifier[dtype]()
    _ = ridge.fit(features, labels)
    same(pipe.transformer.components_, rbm.components_)
    var hidden = rbm.transform(X)
    same(pipe.predict(X), ridge.predict(hidden))
    assert_equal(pipe.score(X, labels), ridge.score(hidden, labels))
    assert_equal(pipe.n_features_in_(), 8)
    assert_equal(pipe.transformer.get_feature_names_out()[5], "bernoullirbm5")

    # Clone copies configuration only; typed nested set_params then refit.
    var named = Pipeline(BernoulliRBM[dtype](6, n_iter=5, random_state=0), RidgeClassifier[dtype](),
                         transformer_name="rbm", estimator_name="logistic")
    _ = named.fit(X, labels)
    var fresh = clone(named)
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = fresh.n_features_in_()
    var params = fresh.get_params()
    params.transformer.n_components = 3
    params.transformer.learning_rate = 0.06
    _ = fresh.set_params(params)
    _ = fresh.fit(X, labels)
    assert_equal(fresh.transformer.components_.shape[0], 3)
    assert_equal(named.transformer.components_.shape[0], 6)
    var manual = BernoulliRBM[dtype](3, learning_rate=0.06, n_iter=5, random_state=0)
    _ = manual.fit(X)
    same(fresh.transformer.components_, manual.components_)


def union[dtype: DType]() raises:
    var X = binary[dtype](20, 6)
    var u = make_union[dtype=dtype](BernoulliRBM[dtype](3, n_iter=2, random_state=4), PCA[dtype](2))
    var stacked = u.fit_transform(X)
    assert_equal(stacked.shape[1], 5)
    var rbm = BernoulliRBM[dtype](3, n_iter=2, random_state=4)
    var a = rbm.fit_transform(X)
    var pca = PCA[dtype](2)
    var b = pca.fit_transform(X)
    for i in range(20):
        for j in range(3):
            assert_equal(stacked.unsafe_get(i * 5 + j), a.unsafe_get(i * 3 + j))
        for j in range(2):
            assert_equal(stacked.unsafe_get(i * 5 + 3 + j), b.unsafe_get(i * 2 + j))
    var names = u.get_feature_names_out()
    assert_equal(len(names), 5)
    assert_equal(names[0], "bernoullirbm__bernoullirbm0")
    assert_equal(names[4], "pca__pca1")
    assert_equal(u.n_features_in_(), 6)


def main() raises:
    pipelines[DType.float64]()
    pipelines[DType.float32]()
    union[DType.float64]()
    union[DType.float32]()
    print("PASS RBM clone, pipeline (exact against manual steps) and feature union composition")
