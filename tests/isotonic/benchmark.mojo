"""Native timing region; input creation and process startup are excluded."""
from std.sys import argv
from std.time import perf_counter_ns
from moxie.isotonic import IsotonicRegression, isotonic_regression

def measure[dtype: DType](n: Int, loops: Int, mode: String) raises:
    var x = List[Scalar[dtype]]()
    var y = List[Scalar[dtype]]()
    var w = List[Scalar[dtype]]()
    for i in range(n):
        x.append(Scalar[dtype](n - i))
        y.append(Scalar[dtype]((i * 17) % 101))
        w.append(Scalar[dtype](1 + i % 7))
    var checksum: Float64 = 0
    var start = perf_counter_ns()
    for iteration in range(loops):
        y[0] = Scalar[dtype](iteration % 101)
        if mode == "kernel":
            var result = isotonic_regression(y, sample_weight=w.copy())
            checksum += Float64(result[n // 2])
        else:
            var model = IsotonicRegression[dtype](increasing="auto", out_of_bounds="clip")
            _ = model.fit(x, y, sample_weight=w)
            checksum += Float64(model.predict(x)[n // 2])
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) / 1e9)
    print(checksum)

def main() raises:
    var args = argv()
    if args[1] == "float32":
        measure[DType.float32](Int(args[2]), Int(args[3]), args[4])
    else:
        measure[DType.float64](Int(args[2]), Int(args[3]), args[4])
