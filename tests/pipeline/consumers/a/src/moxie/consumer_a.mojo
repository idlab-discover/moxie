"""Test double: shared storage consumer, not an implemented estimator."""
from moxie._pipeline import CompositionStep
from moxie.base import BaseEstimator
from moxie._dense import row_squared_norms
from numojo import NDArray

struct ProbeA(CompositionStep):
    comptime Params = Int
    var config: Int
    var fitted: Bool
    def __init__(out self, config: Int = 1):
        self.config = config
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.config
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        self.config = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.config)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def consume[dtype: DType](self, X: NDArray[dtype]) raises -> NDArray[dtype]:
        return row_squared_norms(X)
