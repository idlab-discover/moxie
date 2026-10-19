"""Test-only training retention probe, not an estimator implementation."""
import numojo as nm
from moxie.gaussian_process._plumbing import retain_training
from moxie._dense import row_squared_norms
from moxie._random import RandomState

@fieldwise_init
struct ConsumerA:
    def inspect[dtype: DType](self, mut X: nm.NDArray[dtype], mut rng: RandomState) raises -> nm.NDArray[dtype]:
        var snapshot = retain_training(X)
        _ = rng.random_sample()
        return row_squared_norms(snapshot)
