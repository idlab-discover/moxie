from moxie.cluster import AgglomerativeClustering
import numojo as nm

def main() raises:
    var model = AgglomerativeClustering()
    var x = nm.zeros[DType.float64](nm.Shape(3,2))
    _ = model.predict(x)
