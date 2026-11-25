"""Batch scalar Float64 validation using decimal-encoded IEEE bits."""

from std.memory import bitcast
from std.sys import argv
from numax.core.libm import exp, log


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as stream:
        data = stream.read()
    for line in data.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split(" ")
        var x = bitcast[DType.float64](Int(String(fields[1])))
        var result = exp(x) if fields[0] == "exp" else log(x)
        print(bitcast[DType.uint64](result))
