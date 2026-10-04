"""Fit empirical and shrunk covariance estimators on a small dataset."""
import numojo as nm
from numojo import Shape
from moxie.covariance import EmpiricalCovariance, ShrunkCovariance, empirical_covariance, log_likelihood


def main() raises:
    var X = nm.zeros[DType.float64](Shape(4, 2))
    var values: List[Float64] = [0.0, 1.0, 2.0, 1.5, 1.0, 0.5, 3.0, 2.5]
    for k in range(8):
        X.unsafe_set(k, values[k])
    var model = EmpiricalCovariance()
    model.fit(X)
    var covariance = model.covariance_()
    print("covariance_:", covariance.unsafe_get(0), covariance.unsafe_get(1), covariance.unsafe_get(3))
    print("score:", model.score(X))
    var distances = model.mahalanobis(X)
    print("squared Mahalanobis of first sample:", distances.unsafe_get(0))
    var shrunk = ShrunkCovariance(shrinkage=0.5)
    shrunk.fit(X)
    print("shrunk covariance_[0, 1]:", shrunk.covariance_().unsafe_get(1))
    print("log_likelihood:", log_likelihood(empirical_covariance(X), model.get_precision()))
