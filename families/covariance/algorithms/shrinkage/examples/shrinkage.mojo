"""Ledoit-Wolf and OAS shrinkage on a small wide dataset."""
import numojo as nm
from numojo import Shape
from moxie.covariance import LedoitWolf, OAS, ledoit_wolf_shrinkage


def main() raises:
    var X = nm.zeros[DType.float64](Shape(5, 4))
    for k in range(20):
        X.unsafe_set(k, Float64((k * 7) % 11) - 5.0)
    var lw = LedoitWolf()
    lw.fit(X)
    print("Ledoit-Wolf shrinkage_:", lw.shrinkage_(), "covariance_[0, 0]:", lw.covariance_().unsafe_get(0))
    print("ledoit_wolf_shrinkage(block_size=2):", ledoit_wolf_shrinkage(X, block_size=2))
    var oas = OAS()
    oas.fit(X)
    print("OAS shrinkage_:", oas.shrinkage_(), "score:", oas.score(X))
