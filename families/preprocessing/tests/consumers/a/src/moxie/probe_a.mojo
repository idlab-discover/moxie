"""Estimator-shaped ownership fixture, not a preprocessing algorithm."""
from numojo import NDArray
from moxie._dense import sum_axis2d
from moxie.preprocessing._validation import validate_dense

struct Probe(Copyable, Movable):
    var learned: Optional[NDArray[DType.float64]]
    def __init__(out self):
        self.learned = None
    def fit(mut self, x: NDArray[DType.float64]) raises:
        validate_dense(x)
        self.learned = sum_axis2d(x, 0)
    def total(self) raises -> Float64:
        if not self.learned:
            raise Error("NotFittedError: call fit")
        return self.learned.value().item(0)
    def clone(self) -> Self:
        return Self()
