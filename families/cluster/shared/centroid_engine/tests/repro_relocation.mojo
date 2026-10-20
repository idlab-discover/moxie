"""Minimal strict-parity regression: tied distances with unequal weights."""
import numojo as nm
from moxie.cluster._centroid_engine import assign, update

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    x[nm.Item(0, 0)] = 1
    x[nm.Item(1, 0)] = 1
    var c = nm.zeros[DType.float64](nm.Shape(3, 1))
    var w = nm.ones[DType.float64](nm.Shape(4))
    w[nm.Item(1)] = 2
    var a = assign(x, c, w)
    var u = update(x, a.labels, w, c)
    for j in range(3):
        print(j, u.centers.item(j, 0), u.masses.item(j))
