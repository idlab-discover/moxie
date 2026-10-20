"""Pack a known result; this example does not perform a neighbor search."""
import numojo as nm
from moxie.neighbors import TopKResult

def main() raises:
    var indices: List[Int] = [2]
    var distances = nm.full[DType.float64](nm.Shape(1, 1), 0.5)
    var result = TopKResult[DType.float64](1, 1, indices^, distances^)
    result.validate(3)
    print("Known neighbor index:", result.indices[0])
