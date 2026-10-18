"""Whole-process benchmark workload; fitting, allocation and transform included."""
from std.sys import argv
from moxie.random_projection import GaussianRandomProjection, ProjectionMatrix


def workload[dtype: DType](rows: Int, cols: Int, count: Int) raises:
    var X = ProjectionMatrix[dtype](rows, cols)
    for i in range(len(X.data)):
        X.data[i] = Scalar[dtype](Float64(i % 31 - 15) / 8)
    var model = GaussianRandomProjection[dtype](count, random_state=42)
    var Y = model.fit_transform(X)
    print(Y.data[0], Y.data[len(Y.data)-1])


def main() raises:
    var args = argv()
    if args[1] == "32":
        workload[DType.float32](Int(args[2]), Int(args[3]), Int(args[4]))
    else:
        workload[DType.float64](Int(args[2]), Int(args[3]), Int(args[4]))
