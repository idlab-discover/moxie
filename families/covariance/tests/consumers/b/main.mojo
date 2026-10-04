import numojo as nm
from moxie import ProbeA, ProbeB
from std.testing import assert_equal

def check[dtype: DType]() raises:
    var x = nm.ones[dtype](nm.Shape(2, 2))
    var a = ProbeA.snapshot(x)
    var b = ProbeB.snapshot(x)
    var shared_a = ProbeA.share(x)
    var shared_b = ProbeB.share(shared_a)
    x.unsafe_set(0, 3)
    assert_equal(a.covariance_.value().unsafe_get(0), 1)
    assert_equal(b.covariance_.value().unsafe_get(0), 1)
    var an = ProbeA.norms(shared_b)
    var bn = ProbeB.norms(shared_a)
    assert_equal(an.unsafe_get(0), 10)
    assert_equal(bn.unsafe_get(0), 10)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: composed consumers use shared state and dense kernels")
