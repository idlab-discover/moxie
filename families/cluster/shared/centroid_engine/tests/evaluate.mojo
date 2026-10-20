"""Deterministic native outputs for local-Cython differential tests."""
import numojo as nm
from std.sys import argv
from moxie.cluster._centroid_engine import assign, update, distances, inertia, center_shift, minibatch_update, ElkanBounds

def emit[dtype: DType](name: String, data: nm.NDArray[dtype]) raises:
    print("shape", name, data.ndim, data.size)
    var flat = data.flatten()
    for i in range(flat.size):
        print(name, i, flat.item(i))

def evaluate[dtype: DType](n: Int, d: Int, k: Int, mode: Int, layout: Int) raises:
    var owner = nm.zeros[dtype](nm.Shape(n, d * 2))
    for i in range(n):
        for f in range(d * 2):
            owner[nm.Item(i, f)] = SIMD[dtype, 1]((i * 17 + f * 7) % 67 - 33) / 8
    var x = owner.view_with_layout(nm.Shape(n, d), nm.NDArrayStrides(d * 2, 2), 0)
    if layout == 1:
        x = owner.view_with_layout(nm.Shape(n, d), nm.NDArrayStrides(d * 2, -2), d * 2 - 1)
    if mode == 2:
        x = nm.full[dtype](nm.Shape(n, d), 2)
    var c = nm.zeros[dtype](nm.Shape(k, d))
    for j in range(k):
        for f in range(d):
            c[nm.Item(j, f)] = SIMD[dtype, 1]((j * 19 + f * 11) % 43 - 21) / 8
            if mode == 1 or mode == 2:
                c[nm.Item(j, f)] = x.item(0, f)
    var w_owner = nm.ones[dtype](nm.Shape(n * 2))
    var w = w_owner.view_with_layout(nm.Shape(n), nm.NDArrayStrides(2), 0)
    for i in range(n):
        w[nm.Item(i)] = SIMD[dtype, 1]((i * 3) % 7) / 2
        if mode == 3:
            w[nm.Item(i)] = 0
        elif mode == 4:
            w[nm.Item(i)] = 1
    emit("x", x)
    emit("weights", w)
    emit("initial", c)
    emit("squared", distances(x, c))
    emit("distance", distances(x, c, False))
    var bounds = ElkanBounds[dtype](x, c)
    emit("upper_init", bounds.upper)
    emit("lower_init", bounds.lower)
    emit("elabels_init", bounds.labels)
    var batch_c = c.copy()
    var batch_m = nm.zeros[dtype](nm.Shape(k))
    for step in range(4):
        var prefix = String(step)
        var result = assign(x, c, w)
        emit(prefix + "_labels", result.labels)
        print(prefix + "_inertia", 0, result.inertia)
        print(prefix + "_single", 0, inertia(x, c, w, result.labels, 0))
        var elkan = bounds.assign(x, w)
        emit(prefix + "_elabels", elkan.labels)
        emit(prefix + "_upper", bounds.upper)
        emit(prefix + "_lower", bounds.lower)
        var eu = update(x, elkan.labels, w, bounds.centers)
        emit(prefix + "_ecenters", eu.centers)
        emit(prefix + "_emasses", eu.masses)
        bounds.advance(eu.centers)
        emit(prefix + "_upper_advanced", bounds.upper)
        emit(prefix + "_lower_advanced", bounds.lower)
        var u = update(x, result.labels, w, c)
        emit(prefix + "_centers", u.centers)
        emit(prefix + "_masses", u.masses)
        emit(prefix + "_shift", center_shift(c, u.centers))
        c = u.centers.copy()
        var ba = assign(x, batch_c, w)
        var batch = minibatch_update(x, ba.labels, w, batch_c, batch_m)
        emit(prefix + "_batch_centers", batch.centers)
        emit(prefix + "_batch_masses", batch.masses)
        batch_c = batch.centers.copy()
        batch_m = batch.masses.copy()

def main() raises:
    var args = argv()
    if args[1] == "f32":
        evaluate[DType.float32](Int(args[2]), Int(args[3]), Int(args[4]), Int(args[5]), Int(args[6]))
    else:
        evaluate[DType.float64](Int(args[2]), Int(args[3]), Int(args[4]), Int(args[5]), Int(args[6]))
