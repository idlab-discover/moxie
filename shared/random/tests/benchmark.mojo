from moxie._random import RandomState, NativeRandom, _log
from std.math import log as stdlib_log
from std.sys import argv
from std.time import perf_counter_ns

def main() raises:
    var args = argv()
    var mode = args[1]
    var n = Int(args[2])
    var rounds = Int(args[3])
    var rng = RandomState(42)
    var native = NativeRandom(42)
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for _ in range(rounds):
        if mode == "normal":
            for _ in range(n):
                checksum += rng.standard_normal()
        elif mode == "uniform":
            for _ in range(n):
                checksum += rng.random_sample()
        elif mode == "native":
            for _ in range(n):
                checksum += Float64(native.step()[0])
        elif mode == "log_intrinsic":
            for i in range(n):
                checksum += _log(Float64(i + 1) / Float64(n + 1))
        elif mode == "log_stdlib":
            for i in range(n):
                checksum += stdlib_log(Float64(i + 1) / Float64(n + 1))
        elif mode == "permutation":
            var result = rng.permutation(n)
            if n > 0:
                checksum += Float64(result.item(0))
        else:
            raise Error("unknown mode")
    var elapsed = Float64(perf_counter_ns() - start) / 1e9
    print(elapsed)
    print(checksum)
