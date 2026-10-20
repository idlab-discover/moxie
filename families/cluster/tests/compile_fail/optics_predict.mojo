from moxie.cluster import OPTICS
import numojo as nm

def main() raises:
    var model = OPTICS()
    var x = nm.zeros[DType.float64](nm.Shape(5, 1))
    _ = model.predict(x)
