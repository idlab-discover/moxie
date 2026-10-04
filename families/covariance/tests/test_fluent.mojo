"""Current foundation mutation borrows preserve covariance estimator identity."""
from moxie.base import UnsupervisedEstimator
from moxie.covariance import EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS, MinCovDet, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV
from numojo import Shape, zeros
from std.testing import assert_true, assert_false


def generic_fit[E: UnsupervisedEstimator](mut model: E, params: E.Params, X: E.Input) raises where conforms_to(E.Target, Deinitable):
    assert_true(model.set_params(params).fit(X).__sklearn_is_fitted__())
    assert_true(model.__sklearn_is_fitted__())


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(40, 2))
    for i in range(40):
        X.unsafe_set(2 * i, (Float64(i % 7) - 3).cast[dtype]())
        X.unsafe_set(2 * i + 1, (Float64((i * 3) % 11) - 5).cast[dtype]())
    var model0 = EmpiricalCovariance[dtype]()
    generic_fit(model0, model0.get_params(), X)
    ref returned0 = model0.set_params(assume_centered=True)
    returned0.assume_centered = False
    assert_false(model0.assume_centered)
    var model1 = ShrunkCovariance[dtype]()
    generic_fit(model1, model1.get_params(), X)
    ref returned1 = model1.set_params(assume_centered=True)
    returned1.assume_centered = False
    assert_false(model1.assume_centered)
    var model2 = LedoitWolf[dtype]()
    generic_fit(model2, model2.get_params(), X)
    ref returned2 = model2.set_params(assume_centered=True)
    returned2.assume_centered = False
    assert_false(model2.assume_centered)
    var model3 = OAS[dtype]()
    generic_fit(model3, model3.get_params(), X)
    ref returned3 = model3.set_params(assume_centered=True)
    returned3.assume_centered = False
    assert_false(model3.assume_centered)
    var model4 = MinCovDet[dtype](random_state=7)
    generic_fit(model4, model4.get_params(), X)
    ref returned4 = model4.set_params(assume_centered=True)
    returned4.assume_centered = False
    assert_false(model4.assume_centered)
    var model5 = EllipticEnvelope[dtype](random_state=7)
    generic_fit(model5, model5.get_params(), X)
    ref returned5 = model5.set_params(assume_centered=True)
    returned5.assume_centered = False
    assert_false(model5.assume_centered)
    var model6 = GraphicalLasso[dtype]()
    generic_fit(model6, model6.get_params(), X)
    ref returned6 = model6.set_params(assume_centered=True)
    returned6.assume_centered = False
    assert_false(model6.assume_centered)
    var model7 = GraphicalLassoCV[dtype](cv=3)
    generic_fit(model7, model7.get_params(), X)
    ref returned7 = model7.set_params(assume_centered=True)
    returned7.assume_centered = False
    assert_false(model7.assume_centered)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: fluent covariance mutations through concrete and shared traits")
