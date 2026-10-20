"""Estimator-shaped ownership fixture, not a learning algorithm."""
from numojo import NDArray
from moxie.neural_network._plumbing import snapshot_matrix
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie.utils.validation import FittedState

struct ProbeA(Movable):
    var state: FittedState
    var learned: Optional[NDArray[DType.float64]]

    def __init__(out self):
        self.state = FittedState()
        self.learned = None

    def fit(mut self, X: NDArray[DType.float64]) raises:
        var owned = snapshot_matrix(X)
        self.state.record_fit(X.shape[0], X.shape[1])
        self.learned = owned^

    def predict(self, X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
        self.state.check_features(X.shape[1])
        return row_squared_norms(X)

    def clone(self) -> Self:
        return Self()

    def draw(self, mut rng: RandomState) -> Float64:
        return rng.random_sample()
