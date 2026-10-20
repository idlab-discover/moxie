"""Native contracts for the family's dense preprocessing/intercept/prediction owner."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isnan, sqrt
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.linear_model._linear_base import (
    preprocess_data,
    set_intercept,
    linear_decision,
    public_coef,
    canonical_coef,
    public_prediction,
    r2_score,
    target_matrix,
    check_predict_input,
    DenseCentering,
    DenseDecision,
    gram_matrix,
    outer_gram,
    transpose_product,
)


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def vector[dtype: DType](values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def tiny_case[dtype: DType]() raises:
    # Handoff fixture: X=[[1],[3]], y=[3,7], weights=[1,1].
    var X = matrix[dtype](2, 1, [1.0, 3.0])
    var y = vector[dtype]([3.0, 7.0])
    var w = vector[dtype]([1.0, 1.0])
    var data = preprocess_data(X, y, Optional(w.copy()), True)
    assert_equal(Float64(data.X_offset.unsafe_get(0)), 2.0)
    assert_equal(Float64(data.y_offset.unsafe_get(0)), 5.0)
    assert_equal(Float64(data.X.unsafe_get(0)), -1.0)
    assert_equal(Float64(data.X.unsafe_get(1)), 1.0)
    assert_equal(Float64(data.y.unsafe_get(0)), -2.0)
    assert_equal(Float64(data.y.unsafe_get(1)), 2.0)
    assert_equal(Float64(data.X_scale.unsafe_get(0)), 1.0)
    assert_true(data.vector_target)
    assert_equal(data.y.ndim, 2)
    assert_equal(Float64(data.sample_weight_sqrt.value().unsafe_get(1)), 1.0)
    var coef = matrix[dtype](1, 1, [2.0])
    var intercept = set_intercept(coef, data.X_offset, data.y_offset, data.X_scale, True)
    assert_equal(Float64(intercept.unsafe_get(0)), 1.0)
    # Inputs are preserved.
    assert_equal(Float64(X.unsafe_get(0)), 1.0)
    assert_equal(Float64(y.unsafe_get(1)), 7.0)
    # Prediction: X=[[3]], coef=[[2]], intercept=[1] -> [[7]].
    var decision = linear_decision(matrix[dtype](1, 1, [3.0]), matrix[dtype](1, 1, [2.0]), vector[dtype]([1.0]))
    assert_equal(decision.ndim, 2)
    assert_equal(Float64(decision.unsafe_get(0)), 7.0)


def weighted_case[dtype: DType]() raises:
    # Weighted offsets: weights [1, 3] -> mean = (1*1 + 3*5)/4 = 4.
    var X = matrix[dtype](2, 2, [1.0, 2.0, 5.0, 6.0])
    var y = matrix[dtype](2, 2, [0.0, 4.0, 8.0, 8.0])
    var w = vector[dtype]([1.0, 3.0])
    var data = preprocess_data(X, y, Optional(w.copy()), True)
    assert_equal(Float64(data.X_offset.unsafe_get(0)), 4.0)
    assert_equal(Float64(data.X_offset.unsafe_get(1)), 5.0)
    assert_equal(Float64(data.y_offset.unsafe_get(0)), 6.0)
    assert_equal(Float64(data.y_offset.unsafe_get(1)), 7.0)
    assert_false(data.vector_target)
    # Row 1 centered then scaled by sqrt(3).
    var root3 = sqrt(Float64(3))
    assert_almost_equal(Float64(data.X.unsafe_get(2)), 1.0 * root3, atol=1e-6)
    assert_almost_equal(Float64(data.y.unsafe_get(3)), 1.0 * root3, atol=1e-6)
    # Row 0 scaled by 1.
    assert_equal(Float64(data.X.unsafe_get(0)), -3.0)
    # No rescaling: centered only, no sqrt weights.
    var plain = preprocess_data(X, y, Optional(w.copy()), True, rescale_with_sw=False)
    assert_false(Bool(plain.sample_weight_sqrt))
    assert_equal(Float64(plain.X.unsafe_get(2)), 1.0)
    # No intercept: zero offsets, rescaling still applies.
    var raw = preprocess_data(X, y, Optional(w.copy()), False)
    assert_equal(Float64(raw.X_offset.unsafe_get(0)), 0.0)
    assert_equal(Float64(raw.y_offset.unsafe_get(1)), 0.0)
    assert_almost_equal(Float64(raw.X.unsafe_get(2)), 5.0 * root3, atol=1e-5)
    # Unweighted: sklearn's None branch.
    var none = preprocess_data(X, y, None, True)
    assert_false(Bool(none.sample_weight_sqrt))
    assert_equal(Float64(none.X_offset.unsafe_get(0)), 3.0)
    # Signed weights are rejected only when rescaling.
    var signed = vector[dtype]([2.0, -1.0])
    with assert_raises():
        _ = preprocess_data(X, y, Optional(signed.copy()), True)
    var centered = preprocess_data(X, y, Optional(signed.copy()), True, rescale_with_sw=False)
    assert_equal(Float64(centered.X_offset.unsafe_get(0)), -3.0)
    # Multi-output intercept reconstruction: coef (2, 2).
    var coef = matrix[dtype](2, 2, [1.0, 0.0, 0.0, 1.0])
    var intercept = set_intercept(coef, data.X_offset, data.y_offset, data.X_scale, True)
    assert_equal(Float64(intercept.unsafe_get(0)), 2.0)
    assert_equal(Float64(intercept.unsafe_get(1)), 2.0)
    var zero = set_intercept(coef, data.X_offset, data.y_offset, data.X_scale, False)
    assert_equal(Float64(zero.unsafe_get(1)), 0.0)


def strided_and_shapes[dtype: DType]() raises:
    # A reversed-row view of [[1,2],[3,4],[5,6]] is [[5,6],[3,4],[1,2]].
    var base = matrix[dtype](3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var view = base.view_with_layout(Shape(3, 2), nm.NDArrayStrides(-2, 1), 4)
    var y = vector[dtype]([1.0, 2.0, 3.0])
    var data = preprocess_data(view, y, None, False)
    assert_equal(Float64(data.X.unsafe_get(0)), 5.0)
    assert_equal(Float64(data.X.unsafe_get(5)), 2.0)
    var coef = public_coef(matrix[dtype](1, 2, [1.0, 2.0]), True)
    assert_equal(coef.ndim, 1)
    assert_equal(coef.shape[0], 2)
    var back = canonical_coef(coef)
    assert_equal(back.ndim, 2)
    assert_equal(back.shape[1], 2)
    var decision = linear_decision(view, back, vector[dtype]([0.5]))
    assert_equal(Float64(decision.unsafe_get(0)), 17.5)
    var prediction = public_prediction(decision, True)
    assert_equal(prediction.ndim, 1)
    var multi = public_prediction(decision, False)
    assert_equal(multi.ndim, 2)
    var t = target_matrix(y)
    assert_equal(t.shape[1], 1)
    with assert_raises():
        _ = linear_decision(view, matrix[dtype](1, 3, [1.0, 1.0, 1.0]), vector[dtype]([0.0]))
    with assert_raises():
        _ = public_coef(matrix[dtype](2, 2, [1.0, 2.0, 3.0, 4.0]), True)
    with assert_raises():
        check_predict_input(view, 3, "Probe")
    check_predict_input(view, 2, "Probe")


def validation[dtype: DType]() raises:
    var X = matrix[dtype](2, 1, [1.0, 3.0])
    var y = vector[dtype]([3.0, 7.0])
    var bad = matrix[dtype](2, 1, [1.0, Float64("nan")])
    with assert_raises():
        _ = preprocess_data(bad, y, None, True)
    with assert_raises():
        _ = preprocess_data(X, vector[dtype]([1.0, Float64("inf")]), None, True)
    with assert_raises():
        _ = preprocess_data(X, vector[dtype]([1.0, 2.0, 3.0]), None, True)
    with assert_raises():
        _ = preprocess_data(X, y, Optional(vector[dtype]([0.0, 0.0])), True)
    with assert_raises():
        _ = preprocess_data(X, y, Optional(vector[dtype]([1.0])), True)


def scores[dtype: DType]() raises:
    var y = vector[dtype]([1.0, 2.0, 3.0])
    assert_equal(r2_score(y, y), 1.0)
    var mean = vector[dtype]([2.0, 2.0, 2.0])
    assert_equal(r2_score(y, mean), 0.0)
    # Weighted: weights [1, 0, 1] ignore the middle sample.
    var w = vector[dtype]([1.0, 0.0, 1.0])
    var wrong_middle = vector[dtype]([1.0, 9.0, 3.0])
    assert_equal(r2_score(y, wrong_middle, Optional(w.copy())), 1.0)
    # Constant truth: perfect prediction scores 1, imperfect scores 0.
    var constant = vector[dtype]([1.0, 1.0])
    assert_equal(r2_score(constant, constant), 1.0)
    assert_equal(r2_score(constant, vector[dtype]([1.0, 2.0])), 0.0)
    assert_true(isnan(r2_score(vector[dtype]([1.0]), vector[dtype]([1.0]))))
    with assert_raises(contains="sum to zero"):
        _ = r2_score(y, y, Optional(vector[dtype]([1.0, -1.0, 0.0])))
    with assert_raises():
        _ = r2_score(y, vector[dtype]([1.0, Float64("nan"), 3.0]))
    # Uniform average over outputs.
    var Y = matrix[dtype](3, 2, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    var P = matrix[dtype](3, 2, [1.0, 2.0, 2.0, 2.0, 3.0, 2.0])
    assert_almost_equal(r2_score(Y, P), 0.5, atol=1e-12)


def products[dtype: DType]() raises:
    var X = matrix[dtype](3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var G = gram_matrix(X)
    assert_equal(Float64(G.unsafe_get(0)), 35.0)
    assert_equal(Float64(G.unsafe_get(1)), 44.0)
    assert_equal(Float64(G.unsafe_get(2)), 44.0)
    assert_equal(Float64(G.unsafe_get(3)), 56.0)
    var K = outer_gram(X)
    assert_equal(K.shape[0], 3)
    assert_equal(Float64(K.unsafe_get(1)), 11.0)
    assert_equal(Float64(K.unsafe_get(8)), 61.0)
    var v = transpose_product(X, vector[dtype]([1.0, 0.0, 1.0]))
    assert_equal(v.ndim, 1)
    assert_equal(Float64(v.unsafe_get(1)), 8.0)
    var M = transpose_product(X, matrix[dtype](3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert_equal(M.ndim, 2)
    assert_equal(Float64(M.unsafe_get(1)), 8.0)
    assert_equal(Float64(M.unsafe_get(3)), 10.0)
    with assert_raises():
        _ = transpose_product(X, vector[dtype]([1.0, 2.0]))


def providers() raises:
    var X = matrix[DType.float64](2, 1, [1.0, 3.0])
    var y = vector[DType.float64]([3.0, 7.0])
    var provider = DenseCentering[DType.float64]()
    var data = provider.prepare(X, y, nm.ones[DType.float64](Shape(2)), True, True)
    assert_equal(data.X_offset.unsafe_get(0), 2.0)
    var decision = DenseDecision[DType.float64]().decision(X, nm.ones[DType.float64](Shape(1, 1)), nm.zeros[DType.float64](Shape(1)))
    assert_equal(decision.unsafe_get(1), 3.0)


def main() raises:
    tiny_case[DType.float32]()
    tiny_case[DType.float64]()
    weighted_case[DType.float32]()
    weighted_case[DType.float64]()
    strided_and_shapes[DType.float32]()
    strided_and_shapes[DType.float64]()
    validation[DType.float32]()
    validation[DType.float64]()
    scores[DType.float32]()
    scores[DType.float64]()
    products[DType.float32]()
    products[DType.float64]()
    providers()
    print("family preprocessing, intercept, prediction and score contracts passed")
