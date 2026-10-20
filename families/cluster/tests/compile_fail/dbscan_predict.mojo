import numojo as nm
from moxie.cluster import DBSCAN
def main() raises:
    var model = DBSCAN()
    var x = nm.zeros[DType.float64](nm.Shape(2,1))
    _ = model.fit(x)
    _ = model.predict(x)
