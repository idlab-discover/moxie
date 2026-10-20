from numojo import NDArray
from moxie.semi_supervised._contracts import DenseAffinity
struct Invalid(DenseAffinity):
    comptime Input = NDArray[DType.float64]
def main():
    pass
