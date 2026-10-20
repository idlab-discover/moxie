import numojo as nm
from moxie import ConsumerA, ConsumerB
from moxie.neighbors import KNeighborsRegressor, LocalOutlierFactor, KernelDensity
from moxie._random import RandomState
from std.testing import assert_equal, assert_raises

def main() raises:
    var a = ConsumerA()
    var b = ConsumerB()
    with assert_raises():
        _ = a.inspect()
    var x = nm.full[DType.float64](nm.Shape(2, 2), 2)
    a.fit(x)
    b.fit(x)
    x.unsafe_set(0, 9)
    var av = a.inspect()
    var bv = b.inspect()
    assert_equal(av.item(0), 8)
    assert_equal(bv.item(0), 8)
    a.fit(x)
    av = a.inspect()
    assert_equal(av.item(0), 85)
    bv = b.inspect()
    assert_equal(bv.item(0), 8)
    var invalid = nm.zeros[DType.float64](nm.Shape(0, 2))
    with assert_raises(): a.fit(invalid)
    av = a.inspect()
    assert_equal(av.item(0), 85)
    # An owned numerical output crosses project boundaries with the same type.
    var exchanged = av.view_with_layout(nm.Shape(1, 2), nm.NDArrayStrides(2, 1), 0)
    b.fit(exchanged)
    bv = b.inspect()
    assert_equal(bv.item(0), 85 * 85 + 8 * 8)
    var ra = RandomState(42)
    var rb = RandomState(42)
    assert_equal(ra.next_uint32(), rb.next_uint32())
    var target = nm.ones[DType.float64](nm.Shape(2))
    var reg = KNeighborsRegressor(1)
    reg.fit(x,target)
    var pred = reg.predict(x.view())
    assert_equal(Float64(pred.unsafe_get(0)),1)
    var lof = LocalOutlierFactor(1)
    lof.fit(x)
    var kde = KernelDensity()
    kde.fit(x)
    var density = kde.score_samples(x)
    assert_equal(density.shape[0],2)
    print("PASS: two consumers share pinned snapshot, dense and RNG modules")
