"""Solve a dense binary logistic model using explicit engine-level arrays."""
import numojo as nm
from moxie.svm._contracts import LinearOptions
from moxie.svm._liblinear import LibLinear


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        x.unsafe_set(i, Float64(i) - 1.5)
        y.unsafe_set(i, Float64(i // 2))
    var weights = nm.ones[DType.float64](nm.Shape(4))
    var class_weights = nm.ones[DType.float64](nm.Shape(2))
    var options = LinearOptions(0, 1.0, 0.1, 1e-6, True, 1.0, 1000, 42)
    var engine = LibLinear()
    var model = engine.solve(x, y, weights, class_weights, options)
    print("coef:", model.coef)
    print("intercept:", model.intercept)
    print("decision values:", engine.decision_values(x, model))
