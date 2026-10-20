"""Ownership-only estimator fixture; no learning algorithm."""
from numojo import NDArray
from moxie.linear_model._contracts import snapshot, validate_xy

struct ConsumerEstimator(Movable):
    var data: Optional[NDArray[DType.float64]]

    def __init__(out self):
        self.data = None

    def fit(mut self, X: NDArray[DType.float64], y: NDArray[DType.float64]) raises:
        validate_xy(X, y)
        self.data = snapshot(X)

    def first(self) raises -> Float64:
        if not self.data:
            raise Error("NotFittedError: fixture has no snapshot")
        return self.data.value().unsafe_get(0)
