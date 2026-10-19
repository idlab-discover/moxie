"""Public imports, fluent mutation, owned state and real Frozen composition."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_almost_equal
from moxie.base import SupervisedEstimator, clone
from moxie.frozen import FrozenEstimator
from moxie.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD


def fitted[E: SupervisedEstimator](mut model: E, params: E.Params, X: E.Input, y: E.Target) raises:
    _ = model.set_params(params).fit(X, y)


def exercise[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(6, 2))
    var y = nm.zeros[dtype](nm.Shape(6, 1))
    for i in range(6):
        X.unsafe_set(i * 2, Scalar[dtype](i + 1))
        X.unsafe_set(i * 2 + 1, Scalar[dtype]((i * i + 3) % 7))
        y.unsafe_set(i, Scalar[dtype](i * 2 + 3))
    var reg = PLSRegression[dtype](1)
    var can = PLSCanonical[dtype](1)
    var cca = CCA[dtype](1)
    var svd = PLSSVD[dtype](1)
    fitted(reg, reg.get_params(), X, y)
    fitted(can, can.get_params(), X, y)
    fitted(cca, cca.get_params(), X, y)
    fitted(svd, svd.get_params(), X, y)
    assert_equal(reg.set_params(n_components=1).fit(X, y).predict(X).shape[0], 6)
    assert_equal(can.set_params(algorithm="svd").fit(X, y).transform(X).shape[1], 1)
    assert_equal(cca.set_params(scale=False).fit(X, y).transform(X, y).x_scores.shape[0], 6)
    assert_equal(svd.set_params(scale=False).fit(X, y).transform(X).shape[1], 1)
    var cloned = clone(svd)
    assert_false(cloned.__sklearn_is_fitted__())
    var copied = svd.copy()
    var original = svd.x_weights_.unsafe_get(0)
    copied.x_weights_.unsafe_set(0, 999.0)
    assert_equal(svd.x_weights_.unsafe_get(0), original)
    var expected = reg.predict(X)
    var frozen = FrozenEstimator(reg^)
    var predicted = frozen.fit(X, y).predict(X)
    assert_almost_equal(predicted.unsafe_get(0), expected.unsafe_get(0), atol=0.0)
    var frozen_svd = FrozenEstimator(svd^)
    assert_equal(frozen_svd.fit(X, y).transform(X).shape[1], 1)
    _ = can.fit_inplace(X, y).set_params(scale=True)
    _ = cca.fit_inplace(X, y).set_params(scale=True)
    _ = cloned.fit_inplace(X, y).set_params(scale=True)


def main() raises:
    exercise[DType.float64]()
    exercise[DType.float32]()
    print("PASS: all public estimators, fluent traits and Frozen composition")
