"""All sixteen sklearn names share one concrete covariance namespace."""
from moxie.covariance import (
    EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS,
    MinCovDet, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV,
    empirical_covariance, shrunk_covariance, log_likelihood,
    ledoit_wolf, ledoit_wolf_shrinkage, oas, fast_mcd, graphical_lasso,
)
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal


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
    var covariance = empirical_covariance(X)
    _ = shrunk_covariance(covariance)
    _ = log_likelihood(covariance, covariance)
    _ = ledoit_wolf(X)
    _ = ledoit_wolf_shrinkage(X)
    _ = oas(X)
    _ = fast_mcd(X, random_state=7)
    _ = graphical_lasso(covariance, 0.1)
    print("PASS: all 16 covariance exports")
