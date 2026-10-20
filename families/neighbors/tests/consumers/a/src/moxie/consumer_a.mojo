"""Estimator-shaped ownership probe, not an implemented estimator."""
import numojo as nm
from moxie._dense import row_squared_norms
from moxie.neighbors import fit_snapshot

struct ConsumerA(Movable):
    var fitted: Optional[nm.NDArray[DType.float64]]

    def __init__(out self):
        self.fitted = None

    def fit(mut self, x: nm.NDArray[DType.float64]) raises:
        self.fitted = fit_snapshot(x)

    def inspect(self) raises -> nm.NDArray[DType.float64]:
        if not self.fitted:
            raise Error("NotFittedError: probe is unfitted")
        return row_squared_norms(self.fitted.value())
