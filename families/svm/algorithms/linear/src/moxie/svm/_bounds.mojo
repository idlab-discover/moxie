# Adapted from sklearn/svm/_bounds.py; BSD-3-Clause.
"""Dense l1_min_c using the accepted matrix product and shared axis reduction."""
import numojo as nm
from std.math import abs, isfinite, max
from moxie._dense import sum_axis2d
from moxie.svm._contracts import Dense, validate_matrix
from moxie.svm._linear import _labels, _codes


def l1_min_c[Label: Copyable & Comparable & Deinitable](
    X: Dense, y: List[Label], *, loss: String = "squared_hinge",
    fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0,
) raises -> Float64:
    validate_matrix(X)
    if len(y) != X.shape[0]:
        raise Error("ValueError: inconsistent target length")
    if loss != "squared_hinge" and loss != "log":
        raise Error("ValueError: loss must be squared_hinge or log")
    if not isfinite(intercept_scaling) or intercept_scaling <= 0:
        raise Error("ValueError: intercept_scaling must be positive and finite")
    var classes = _labels(y)
    var codes = _codes(y, classes)
    var rows = len(classes)
    if rows <= 2:
        rows = 1
    var signs = nm.full[DType.float64](nm.Shape(rows, X.shape[0]), -1.0)
    for i in range(X.shape[0]):
        if len(classes) == 2:
            if codes.unsafe_get(i) == 1:
                signs.unsafe_set(i, 1.0)
        elif len(classes) > 2:
            signs.unsafe_set(Int(codes.unsafe_get(i)) * X.shape[0] + i, 1.0)
    var product = signs @ X
    var denominator = Float64(0)
    for i in range(product.size):
        denominator = max(denominator, abs(product.unsafe_get(i)))
    if fit_intercept:
        var sums = sum_axis2d(signs, 1)
        for i in range(rows):
            denominator = max(denominator, abs(sums.unsafe_get(i) * intercept_scaling))
    if denominator == 0:
        raise Error("ValueError: ill-posed l1_min_c: all coefficients always zero")
    if loss == "log":
        return 2.0 / denominator
    return 0.5 / denominator


def l1_min_c[Label: Copyable & Comparable & Deinitable](
    X: nm.NDArray[DType.float32], y: List[Label], *, loss: String = "squared_hinge",
    fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0,
) raises -> Float64:
    return l1_min_c(X.astype[DType.float64](), y, loss=loss,
                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
