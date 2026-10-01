from moxie.pipeline import FeatureUnion, UnionSentinel
from numojo import zeros, Shape

def main() raises:
    var union = FeatureUnion(UnionSentinel("raw"))
    _ = union.predict(zeros[DType.float64](Shape(1, 1)))
