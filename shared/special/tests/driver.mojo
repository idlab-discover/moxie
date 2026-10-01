"""Validation-only batch text interchange; no Python in the runtime path."""
from std.sys import argv
from moxie._special import gammainc


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as stream:
        data = stream.read()
    for line in data.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split(" ")
        var a = Float64(String(fields[1]))
        var x = Float64(String(fields[2]))
        try:
            if fields[0] == "32":
                print(Float64(gammainc(Float32(a), Float32(x))))
            else:
                print(gammainc(a, x))
        except error:
            print(String(error))
