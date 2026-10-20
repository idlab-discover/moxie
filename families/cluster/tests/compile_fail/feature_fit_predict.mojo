from moxie.cluster import FeatureAgglomeration
import numojo as nm

def main() raises:
    var model = FeatureAgglomeration()
    var x = nm.zeros[DType.float64](nm.Shape(3,2))
    _ = model.fit_predict(x)
