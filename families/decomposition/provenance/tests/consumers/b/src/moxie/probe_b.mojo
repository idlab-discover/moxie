"""Estimator-shaped ownership fixture, not a decomposition algorithm."""
from numojo import NDArray, zeros, Shape
from moxie.decomposition._contracts import owned_matrix, validate_matrix
from moxie.utils import FittedState
from moxie._dense import row_squared_norms

struct ProbeB[dtype: DType](Movable):
    var data: NDArray[Self.dtype]
    var state: FittedState

    def __init__(out self) raises:
        self.data = zeros[Self.dtype](Shape(0, 1))
        self.state = FittedState()

    def fit(mut self, x: NDArray[Self.dtype]) raises:
        validate_matrix(x)
        var snapshot = owned_matrix(x)
        self.state.record_fit(x.shape[0], x.shape[1])
        self.data = snapshot^

    def norms(self) raises -> NDArray[Self.dtype]:
        self.state.check_features(self.data.shape[1])
        return row_squared_norms(self.data)
