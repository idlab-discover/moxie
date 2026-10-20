"""Fit BayesianRidge and obtain posterior predictive uncertainty."""
from numojo import Shape, zeros
from moxie.linear_model import BayesianRidge


def main() raises:
    var X = zeros[DType.float64](Shape(6, 1))
    var y = zeros[DType.float64](Shape(6))
    for i in range(6):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Float64(1 + 2 * i) + Float64(i % 2) * 0.2)
    var model = BayesianRidge(compute_score=True)
    _ = model.fit(X, y)
    var result = model.predict_std(X)
    print("Predictions:", result[0])
    print("Standard deviations:", result[1])
    print("Noise precision:", model.alpha_)
