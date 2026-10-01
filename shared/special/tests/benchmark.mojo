"""Changing-input scalar latency; native region excludes process startup."""
from std.sys import argv
from std.time import perf_counter_ns
from moxie._special import gammainc


def main() raises:
    var args = argv()
    var a = Float64(args[1])
    var scale = Float64(args[2])
    var rounds = Int(args[3])
    var checksum = Float64(0)
    var start = perf_counter_ns()
    for i in range(rounds):
        checksum += gammainc(a, a * scale + Float64(i % 2) * 0.01)
    print(Float64(perf_counter_ns() - start) / 1e9)
    print(checksum)
