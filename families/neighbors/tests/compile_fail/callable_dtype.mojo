from moxie.neighbors import NearestNeighbors
from callable_metric import WeightedL1
import numojo as nm

def main() raises:
    var metric = WeightedL1[DType.float32](nm.ones[DType.float64](nm.Shape(2)))
    var model = NearestNeighbors[DType.float64, WeightedL1[DType.float32]](algorithm="brute", metric_params=metric.copy())
