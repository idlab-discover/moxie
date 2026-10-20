"""Assembly test double; no manifold estimator."""
import numojo as nm
from moxie.manifold._plumbing import EmbeddingResult
from moxie._dense import row_squared_norms
from moxie._random import RandomState

def snapshot[dtype: DType](x: nm.NDArray[dtype]) raises -> EmbeddingResult[dtype]:
    return EmbeddingResult(x, x.shape[0], x.shape[1])

def measure[dtype: DType](x: nm.NDArray[dtype]) raises -> Float64:
    var norms = row_squared_norms(x)
    var rng = RandomState(42)
    return Float64(norms.unsafe_get(0)) + rng.random_sample()

struct EstimatorProbe(Movable):
    """Estimator-shaped assembly fixture, not a learning algorithm."""
    var fitted: Bool
    var signature_: Float64

    def __init__(out self):
        self.fitted = False
        self.signature_ = 0

    def fit[dtype: DType](mut self, x: nm.NDArray[dtype]) raises:
        var owned = snapshot(x)
        var signature = measure(owned.embedding)
        self.signature_ = signature
        self.fitted = True

    def __sklearn_clone__(self) -> Self:
        return Self()
