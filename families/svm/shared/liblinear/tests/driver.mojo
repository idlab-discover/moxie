"""Text protocol for validation only; Python never runs inside the engine."""
from std.sys import argv
import numojo as nm
from moxie.svm._contracts import LinearOptions
from moxie.svm._liblinear import LibLinear


def main() raises:
    var args = argv()
    var contents: String
    with open(args[1], "r") as file:
        contents = file.read()
    var tokens = contents.split()
    var n = Int(tokens[0])
    var p = Int(tokens[1])
    var k = Int(tokens[2])
    var solver = Int(tokens[3])
    var options = LinearOptions(solver, Float64(tokens[4]), Float64(tokens[5]), Float64(tokens[6]), Int(tokens[7]) != 0, Float64(tokens[8]), Int(tokens[9]), Int(tokens[10]))
    var x = nm.zeros[DType.float64](nm.Shape(n, p))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var sw = nm.zeros[DType.float64](nm.Shape(n))
    var cw = nm.zeros[DType.float64](nm.Shape(k))
    var index = 11
    for i in range(n * p):
        x.unsafe_set(i, Float64(tokens[index]))
        index += 1
    for i in range(n):
        y.unsafe_set(i, Float64(tokens[index]))
        index += 1
    for i in range(n):
        sw.unsafe_set(i, Float64(tokens[index]))
        index += 1
    for i in range(k):
        cw.unsafe_set(i, Float64(tokens[index]))
        index += 1
    var engine = LibLinear()
    var model = engine.solve(x, y, sw, cw, options)
    print(model.coef.shape[0], model.coef.shape[1], model.fit_status)
    for i in range(model.coef.shape[0] * p):
        print(model.coef.unsafe_get(i))
    for i in range(model.intercept.shape[0]):
        print(model.intercept.unsafe_get(i))
    for i in range(model.iterations.shape[0]):
        print(model.iterations.unsafe_get(i))
