import numojo as nm
from moxie.cluster import AffinityPropagation

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        x[nm.Item(i, 0)] = 1.0 if i < 3 else 4.0
        x[nm.Item(i, 1)] = 2.0 if i % 3 == 0 else (4.0 if i % 3 == 1 else 0.0)
    var model = AffinityPropagation(random_state=5)
    _ = model.fit(x)
    for i in range(6):
        print(model.labels_.item(i))
