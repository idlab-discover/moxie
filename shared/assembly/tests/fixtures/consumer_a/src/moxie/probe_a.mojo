"""Estimator-shaped assembly test double, not a sklearn estimator port."""
from moxie.base import BaseEstimator, clone
from moxie._dense import row_squared_norms
from moxie.utils.validation import FittedState, check_is_fitted
from numojo import NDArray
from std.testing import assert_equal, assert_raises


struct ProbeA[dtype: DType](BaseEstimator):
    comptime Params = Int
    var scale: Int
    var state: FittedState

    def __init__(out self, scale: Int = 1):
        self.scale = scale
        self.state = FittedState()

    def get_params(self, deep: Bool = True) raises -> Int:
        return self.scale

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        self.scale = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.scale)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state.is_fitted()

    def fit[
        origin: MutOrigin,
    ](
        ref[origin] self,
        data: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        self.state.record_fit(data.shape[0], data.shape[1])
        return self

    def predict(self, data: NDArray[Self.dtype]) raises -> Float64:
        check_is_fitted(self)
        self.state.check_features(data.shape[1])
        var norms = row_squared_norms(data)
        return Float64(norms.item(0)) * Float64(self.scale)


def evaluate_a[dtype: DType](data: NDArray[dtype]) raises -> Float64:
    var estimator = ProbeA[dtype]()
    with assert_raises():
        _ = estimator.predict(data)
    _ = estimator.fit(data)
    _ = estimator.fit(data)
    var fresh = clone(estimator)
    assert_equal(fresh.__sklearn_is_fitted__(), False)
    _ = fresh.set_params(2)
    _ = fresh.fit(data)
    assert_equal(fresh.predict(data), estimator.predict(data) * 2)
    assert_equal(estimator.get_params(), 1)
    return estimator.predict(data)
