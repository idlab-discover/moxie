from moxie.cross_decomposition import PLSSVD
import numojo as nm

def main() raises:
    var model = PLSSVD()
    var X = nm.zeros[DType.float64](nm.Shape(3, 2))
    _ = model.predict(X)
