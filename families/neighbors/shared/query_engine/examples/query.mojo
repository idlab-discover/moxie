"""Explicit dense Euclidean neighbor queries; no estimator wrapper."""
import numojo as nm
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import BruteForce

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(3,1))
    x.unsafe_set(1,2)
    x.unsafe_set(2,5)
    var engine = BruteForce(Metric[DType.float64]("euclidean"))
    engine.fit(x)
    var neighbors = engine.kneighbors(None,1)
    for i in range(3):
        print("row",i,"nearest other row",neighbors.indices[i],"distance",neighbors.distances.value().unsafe_get(i))
