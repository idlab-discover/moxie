"""All sixteen sklearn names share one concrete covariance namespace."""
from moxie.covariance import (
    CovarianceEstimator,
    EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS,
    MinCovDet, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV,
    empirical_covariance, shrunk_covariance, log_likelihood,
    ledoit_wolf, ledoit_wolf_shrinkage, oas, fast_mcd, graphical_lasso,
)
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal


def covariance_width[E: CovarianceEstimator](estimator: E) raises -> Int:
    comptime assert E.Input == NDArray[DType.float64]
    var covariance = rebind_var[NDArray[DType.float64]](estimator.covariance_())
    return covariance.shape[1]


def main() raises:
    var X = zeros[DType.float64](Shape(20, 2))
    for i in range(20):
        X.unsafe_set(2 * i, Float64(i % 7))
        X.unsafe_set(2 * i + 1, Float64(i % 3))
    var empirical = EmpiricalCovariance()
    var shrunk = ShrunkCovariance()
    var lw = LedoitWolf()
    var oa = OAS()
    var robust = MinCovDet(random_state=7)
    var envelope = EllipticEnvelope(random_state=7)
    var graphical = GraphicalLasso()
    var cv = GraphicalLassoCV()
    empirical.fit(X)
    shrunk.fit(X)
    lw.fit(X)
    oa.fit(X)
    robust.fit(X)
    envelope.fit(X)
    graphical.fit(X)
    cv.fit(X)
    assert_equal(cv.n_features_in_(), 2)
    assert_equal(covariance_width(empirical), 2)
    assert_equal(covariance_width(shrunk), 2)
    assert_equal(covariance_width(lw), 2)
    assert_equal(covariance_width(oa), 2)
    assert_equal(covariance_width(robust), 2)
    assert_equal(covariance_width(graphical), 2)
    assert_equal(covariance_width(cv), 2)
    var covariance = empirical_covariance(X)
    _ = shrunk_covariance(covariance)
    _ = log_likelihood(covariance, covariance)
    _ = ledoit_wolf(X)
    _ = ledoit_wolf_shrinkage(X)
    _ = oas(X)
    _ = fast_mcd(X, random_state=7)
    _ = graphical_lasso(covariance, 0.1)
    print("PASS: all 16 covariance exports")
