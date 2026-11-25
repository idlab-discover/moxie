"""Evaluate runtime operands from their signed binary64 bit patterns."""

from std.memory import bitcast
from std.sys import argv
from moxie._dense import pow_float64


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as stream:
        data = stream.read()
    for line in data.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split(" ")
        var base = bitcast[DType.float64](Int(String(fields[0])))
        var exponent = bitcast[DType.float64](Int(String(fields[1])))
        print(bitcast[DType.uint64](pow_float64(base, exponent)))
