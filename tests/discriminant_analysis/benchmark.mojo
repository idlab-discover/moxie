"""Timing driver: benchmark MODE N P LOOPS -> mean seconds per fit+predict_proba.

MODE is lda-svd, lda-lsqr, lda-eigen, qda-svd or qda-eigen. Data: three classes with
deterministic pseudo-random features (LCG seed 42), identical to tools/benchmark.py.
"""
from std.sys import argv
from std.time import perf_counter_ns
from numojo import NDArray, Shape, zeros
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def dataset(n: Int, p: Int) raises -> Tuple[NDArray[DType.float64], List[Int]]:
    var X = zeros[DType.float64](Shape(n, p))
    var y = List[Int](capacity=n)
    var state: UInt64 = 42
    for i in range(n):
        var label = i % 3
        y.append(label)
        for j in range(p):
            state = state * 6364136223846793005 + 1442695040888963407
            var u = Float64(state >> 11) / 9007199254740992.0
            X.unsafe_set(i * p + j, u + Float64(label) * (0.5 if j % 2 == 0 else 0.1))
    return (X^, y^)


def main() raises:
    var args = argv()
    var mode = String(args[1])
    var n = Int(String(args[2]))
    var p = Int(String(args[3]))
    var loops = Int(String(args[4]))
    var data = dataset(n, p)
    var start = perf_counter_ns()
    var checksum: Float64 = 0
    for _ in range(loops):
        if mode.startswith("lda-"):
            var model = LinearDiscriminantAnalysis(solver=String(mode[byte=4:]))
            model.fit(data[0], data[1])
            checksum += Float64(model.predict_proba(data[0]).unsafe_get(0))
        else:
            var model = QuadraticDiscriminantAnalysis(solver=String(mode[byte=4:]))
            model.fit(data[0], data[1])
            checksum += Float64(model.predict_proba(data[0]).unsafe_get(0))
    var elapsed = Float64(perf_counter_ns() - start) / 1e9
    print(elapsed / Float64(loops))
    print("checksum", checksum)
