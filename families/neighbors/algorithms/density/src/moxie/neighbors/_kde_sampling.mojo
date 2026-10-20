"""Tophat draws using sklearn's Gaussian-radius CDF transformation."""
import numojo as nm
from std.math import sqrt, pow
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie._special import gammainc
from moxie.neighbors._distance import at


def sample_tophat[dtype: DType](
    data: nm.NDArray[dtype], indices: List[Int], bandwidth: Float64,
    mut rng: RandomState,
) raises -> nm.NDArray[DType.float64]:
    var dim = data.shape[1]
    var samples = nm.zeros[DType.float64](nm.Shape(len(indices), dim))
    for i in range(len(indices)):
        for j in range(dim):
            samples.unsafe_set(i * dim + j, rng.normal())
    var squared = row_squared_norms(samples)
    for i in range(len(indices)):
        var radius_squared = squared.unsafe_get(i)
        # Preserve the reference expression, including its zero-radius NaN.
        var correction = (
            pow(gammainc(Float64(dim) / 2, radius_squared / 2), Float64(1) / Float64(dim))
            * bandwidth / sqrt(radius_squared)
        )
        for j in range(dim):
            samples.unsafe_set(
                i * dim + j,
                at(data, indices[i], j) + samples.unsafe_get(i * dim + j) * correction,
            )
    return samples^
