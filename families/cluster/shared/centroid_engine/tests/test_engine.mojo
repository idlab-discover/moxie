import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.cluster._centroid_engine import Lloyd, ElkanBounds, distances, center_shift, minibatch_update, inertia, update

def check[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(4, 2))
    x[nm.Item(1, 0)] = 0.5
    x[nm.Item(2, 0)] = 0.5
    x[nm.Item(2, 1)] = 1
    x[nm.Item(3, 0)] = 1
    x[nm.Item(3, 1)] = 1
    var c = nm.zeros[dtype](nm.Shape(2, 2))
    c[nm.Item(1, 0)] = 1
    c[nm.Item(1, 1)] = 1
    var w = nm.ones[dtype](nm.Shape(4))
    w[nm.Item(0)] = 3
    w[nm.Item(3)] = 3
    var engine = Lloyd[dtype]()
    var a = engine.assign(x, c, w)
    assert_equal(a.labels.item(0), 0)
    assert_equal(a.labels.item(1), 0)
    assert_equal(a.labels.item(2), 1)
    assert_equal(a.labels.item(3), 1)
    assert_equal(a.inertia, 0.5)
    var u = engine.update(x, a.labels, w, c)
    assert_equal(u.centers.item(0, 0), 0.125)
    assert_equal(u.centers.item(1, 0), 0.875)
    assert_equal(u.masses.item(0), 4)
    assert_equal(inertia(x, u.centers, w, a.labels), 0.375)
    var bounds = ElkanBounds[dtype](x, c)
    bounds.advance(u.centers)
    var e = bounds.assign(x, w)
    assert_equal(e.inertia, 0.375)
    var zero = nm.zeros[dtype](nm.Shape(2))
    var batch = minibatch_update(x, a.labels, w, c, zero)
    var again = minibatch_update(x, a.labels, w, batch.centers, batch.masses)
    assert_equal(again.masses.item(0), 8)
    assert_equal(again.centers.item(0, 0), 0.125)
    assert_equal(zero.item(0), 0)
    var shifts = center_shift(c, u.centers)
    assert_equal(shifts.item(0), 0.125)
    var dist = distances(x, c)
    assert_equal(dist.item(0, 1), 2)
    with assert_raises():
        var bad = nm.zeros[dtype](nm.Shape(3))
        _ = engine.assign(x, c, bad)
    with assert_raises():
        var bad_labels = nm.full[DType.int64](nm.Shape(4), -1)
        _ = engine.update(x, bad_labels, w, c)
    assert_equal(c.item(0, 0), 0)
    print("engine contracts passed", dtype)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    edges[DType.float32]()
    edges[DType.float64]()
    relocation_regression[DType.float32]()
    relocation_regression[DType.float64]()


def edges[dtype: DType]() raises:
    var singleton = nm.full[dtype](nm.Shape(1, 1), 4)
    var two_centers = nm.full[dtype](nm.Shape(2, 1), 4)
    var singleton_weights = nm.full[dtype](nm.Shape(1), 2)
    var singleton_labels = nm.zeros[DType.int64](nm.Shape(1))
    with assert_raises(): _ = update(singleton, singleton_labels, singleton_weights, two_centers)
    var split = update(singleton, singleton_labels, singleton_weights, two_centers, allow_singleton=True)
    assert_equal(split.centers.item(0, 0), 4)
    assert_equal(split.centers.item(1, 0), 4)
    assert_equal(split.masses.item(0), 2)
    assert_equal(split.masses.item(1), 0)
    var engine = Lloyd[dtype]()
    var x = nm.full[dtype](nm.Shape(4, 1), 2)
    var c = nm.full[dtype](nm.Shape(3, 1), 2)
    var weights = nm.ones[dtype](nm.Shape(4))
    var a = engine.assign(x, c, weights)
    for i in range(4):
        assert_equal(a.labels.item(i), 0)
    var u = engine.update(x, a.labels, weights, c)
    for j in range(3):
        assert_equal(u.centers.item(j, 0), 2)
    assert_equal(u.masses.item(1), 0)
    var empty_x = nm.zeros[dtype](nm.Shape(0, 1))
    var empty_w = nm.zeros[dtype](nm.Shape(0))
    var empty_a = engine.assign(empty_x, c, empty_w)
    assert_equal(empty_a.labels.size, 0)
    assert_equal(empty_a.inertia, 0)
    with assert_raises():
        _ = engine.update(empty_x, empty_a.labels, empty_w, c)
    var many_c = nm.zeros[dtype](nm.Shape(5, 1))
    _ = engine.assign(x, many_c, weights)
    with assert_raises():
        _ = engine.update(x, a.labels, weights, many_c)
    with assert_raises():
        var bad = nm.zeros[dtype](nm.Shape(4, 2))
        _ = engine.assign(bad, c, weights)
    with assert_raises():
        var bad = nm.zeros[dtype](nm.Shape(4))
        _ = engine.assign(bad, c, weights)
    from std.utils.numerics import inf
    with assert_raises():
        var bad = x.copy()
        bad[nm.Item(0, 0)] = inf[dtype]()
        _ = engine.assign(bad, c, weights)
    # Results/snapshots are independent from mutable caller inputs.
    var bounds = ElkanBounds[dtype](x, c)
    c[nm.Item(0, 0)] = 100
    assert_equal(bounds.centers.item(0, 0), 2)
    var bound_result = bounds.assign(x, weights)
    bound_result.labels[nm.Item(0)] = 2
    assert_equal(bounds.labels.item(0), 0)
    with assert_raises():
        var bad = nm.zeros[dtype](nm.Shape(3, 1))
        var bad_weights = nm.ones[dtype](nm.Shape(3))
        _ = bounds.assign(bad, bad_weights)
    var old_mass = nm.ones[dtype](nm.Shape(3))
    var no_weight = nm.zeros[dtype](nm.Shape(4))
    var batch = minibatch_update(x, a.labels, no_weight, c, old_mass)
    assert_equal(batch.centers.item(0, 0), 100)
    assert_equal(batch.masses.item(0), 1)
    # Signed weights are accepted at this private seam, as in Cython.
    weights[nm.Item(0)] = -1
    var signed = engine.assign(x, c, weights)
    assert_equal(signed.labels.item(0), 1)
    print("edge contracts passed", dtype)


def relocation_regression[dtype: DType]() raises:
    # Unequal weights make partition order observable even for duplicate rows.
    var x = nm.zeros[dtype](nm.Shape(4, 1))
    x.unsafe_set(0, 1)
    x.unsafe_set(1, 1)
    var centers = nm.zeros[dtype](nm.Shape(3, 1))
    var weights = nm.ones[dtype](nm.Shape(4))
    weights.unsafe_set(1, 2)
    var engine = Lloyd[dtype]()
    var assigned = engine.assign(x, centers, weights)
    var result = engine.update(x, assigned.labels, weights, centers)
    assert_equal(result.masses.item(0), 2)
    assert_equal(result.masses.item(1), 1)
    assert_equal(result.masses.item(2), 2)
    assert_equal(result.centers.item(0, 0), 0)
    assert_equal(result.centers.item(1, 0), 1)
    assert_equal(result.centers.item(2, 0), 1)
    for i in range(4):
        assert_equal(assigned.labels.item(i), 0)
    assert_equal(centers.item(1, 0), 0)
