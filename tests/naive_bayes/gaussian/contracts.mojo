"""Dense Gaussian API, ownership, label and transactional lifecycle contracts."""
import numojo as nm
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.naive_bayes import GaussianNB, NumericDomain, StringDomain
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import dense_from_lists, log_normalize, exp_normalize
from std.math import isnan


@no_inline
def check_float32() raises:
    var X = dense_from_lists([[10000000.0], [10000001.0], [10000003.0],
                              [10000004.0], [10000005.0], [10000006.0]]).astype[DType.float32]()
    var y: List[Int] = [0, 0, 0, 1, 1, 1]
    var model = GaussianNB[dtype=DType.float32](var_smoothing=0)
    _ = model.fit(X, y)
    # Type annotations enforce storage and return dtypes at compile time.
    var means: nm.NDArray[DType.float32] = model.theta_.copy()
    var variances: nm.NDArray[DType.float32] = model.var_.copy()
    var counts: nm.NDArray[DType.float32] = model.class_count_.copy()
    var priors: nm.NDArray[DType.float32] = model.class_prior_.copy()
    var epsilon: Float32 = model.epsilon_
    var joint: nm.NDArray[DType.float32] = model.predict_joint_log_proba(X)
    var logs: nm.NDArray[DType.float32] = model.predict_log_proba(X)
    var probabilities: nm.NDArray[DType.float32] = model.predict_proba(X)
    assert_equal(means.item(0, 0), Float32(10000001))
    # Retained native deviation: default arithmetic remains Float64 even
    # when input storage is Float32; explicit dtype reproduces sklearn rounding.
    var default_model = GaussianNB(var_smoothing=0)
    _ = default_model.fit(X, y)
    var default_variances: nm.NDArray[DType.float64] = default_model.var_.copy()
    assert_true(abs(default_variances.item(0, 0) - 14.0 / 9.0) < 1e-12)
    assert_true(abs(default_variances.item(0, 0) - Float64(variances.item(0, 0))) > 0.1)
    # Widening then casting gives 1.5555556 instead of this reference variance.
    assert_equal(variances.item(0, 0), Float32(5) / 3)
    assert_equal(counts.item(0), Float32(3))
    assert_equal(priors.item(0), Float32(0.5))
    assert_equal(epsilon, Float32(0))
    var copied = model.copy()
    copied.theta_[Item(0, 0)] = 0
    assert_equal(model.theta_.item(0, 0), means.item(0, 0))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var empty: nm.NDArray[DType.float32] = fresh.theta_.copy()
    assert_equal(empty.size, 0)
    _ = fresh.partial_fit(X, y, classes=List[Int]([0, 1]))
    _ = fresh.partial_fit(X, y)
    assert_equal(fresh.class_count_.item(0), Float32(6))
    var reversed_X = X.view_with_layout(Shape(6, 1), nm.NDArrayStrides(-1, 1), 5)
    _ = fresh.fit(reversed_X, List[Int]([1, 1, 1, 0, 0, 0]))
    assert_equal(fresh.predict(reversed_X), List[Int]([1, 1, 1, 0, 0, 0]))
    # Explicit dtype also controls Float64 input conversion and List outputs.
    _ = fresh.fit(X.astype[DType.float64](), y)
    assert_equal(fresh.theta_.item(0, 0), means.item(0, 0))
    var list_probabilities: List[List[Float64]] = fresh.predict_proba([[10000001.0]])
    assert_equal(list_probabilities[0][0], Float64(probabilities.item(1, 0)))
    var bad = X.copy()
    bad[Item(0, 0)] = Float32(Float64("nan"))
    with assert_raises():
        _ = fresh.partial_fit(bad, y)
    assert_equal(fresh.class_count_.item(0), Float32(3))
    var too_large = X.astype[DType.float64]()
    too_large[Item(0, 0)] = 1e100
    with assert_raises():
        _ = fresh.fit(too_large, y)
    with assert_raises():
        _ = fresh.predict(too_large)
    with assert_raises():
        _ = fresh.partial_fit(X, y, sample_weight=1e100)
    with assert_raises():
        _ = fresh.partial_fit(X, y, sample_weight=1e-100)
    assert_equal(fresh.class_count_.item(0), Float32(3))
    _ = fresh.fit(X, List[Int]([1, 1, 1, 1, 1, 1]))
    assert_equal(fresh.classes_, List[Int]([1]))
    var edge32 = nm.zeros[DType.float32](Shape(3, 2))
    edge32[Item(0, 0)] = Float32(Float64("nan"))
    edge32[Item(1, 0)] = -Float32(Float64("inf"))
    edge32[Item(1, 1)] = -Float32(Float64("inf"))
    edge32[Item(2, 0)] = Float32(Float64("inf"))
    var normalized = log_normalize(edge32)
    assert_true(isnan(normalized.item(0, 1)))
    assert_true(isnan(normalized.item(1, 0)))
    assert_true(isnan(normalized.item(2, 0)))
    assert_equal(normalized.item(2, 1), -Float32(Float64("inf")))
    assert_equal(exp_normalize(edge32).item(2, 1), Float32(0))


@no_inline
def check_probability_dtype() raises:
    # Pinned sklearn Float32 normalization loses log(2) at a -2e9 joint
    # score; the retained Float64 default preserves it. Test the visible
    # deviation, not just low-bit differences in learned variance.
    var X = dense_from_lists([[0.0, 1.0], [1.0, 0.0]]).astype[DType.float32]()
    var Q = dense_from_lists([[1.0, 1.0]]).astype[DType.float32]()
    var y: List[Int] = [9, -2]
    var default_model = GaussianNB()
    var reference32 = GaussianNB[dtype=DType.float32]()
    _ = default_model.partial_fit(X, y, classes=y.copy())
    _ = reference32.partial_fit(X, y, classes=y.copy())
    var wide = default_model.predict_proba(Q)
    var narrow = reference32.predict_proba(Q)
    for c in range(2):
        assert_true(abs(wide.item(0, c) - 0.5) < 1e-7)
        assert_equal(narrow.item(0, c), Float32(1))


def main() raises:
    check_probability_dtype()
    check_float32()
    var X = dense_from_lists([[-3.0, 1.0], [-1.0, 2.0], [2.0, 3.0], [4.0, 4.0]])
    var y: List[Int] = [9, 9, -2, -2]
    var model = GaussianNB()
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.fit(X, y)
    assert_equal(model.classes_, List[Int]([-2, 9]))
    assert_equal(model.theta_.shape[0], 2)
    assert_equal(model.theta_.shape[1], 2)
    assert_equal(model.theta_.item(0, 0), 3.0)
    assert_equal(model.class_count_.item(1), 2.0)
    assert_equal(model.score(X, y), 1.0)
    assert_equal(model.predict_proba(X).shape[1], 2)
    assert_equal(model.predict_log_proba(X).shape[0], 4)
    var copied = model.copy()
    copied.theta_[Item(0, 0)] = 123.0
    assert_equal(model.theta_.item(0, 0), 3.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.theta_.size, 0)
    var bad = X.copy()
    bad[Item(0, 0)] = Float64("nan")
    with assert_raises():
        _ = model.fit(bad, y)
    with assert_raises():
        _ = model.partial_fit(X, List[Int]([7, 7, 7, 7]))
    with assert_raises():
        _ = model.partial_fit(X, y, classes=List[Int]([9, -2, 8]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([0.0, 0.0, 0.0, 0.0]))
    with assert_raises():
        _ = model.fit(X, y, sample_weight=List[Float64]([1.0, -1.0, 1.0, 1.0]))
    with assert_raises():
        _ = model.predict(nm.ones[DType.float64](Shape(2, 3)))
    with assert_raises():
        _ = model.predict(nm.ones[DType.float64](Shape(2)))
    with assert_raises():
        _ = model.predict(nm.ones[DType.float64](Shape(0, 2)))
    assert_equal(model.score(X, y), 1.0)
    _ = model.set_params(var_smoothing=-1.0)
    with assert_raises():
        _ = model.fit(X, y)
    assert_equal(model.score(X, y), 1.0)
    _ = model.set_params(var_smoothing=1e-9)
    var params = model.get_params()
    params.priors = List[Float64]([0.2, 0.8])
    _ = model.set_params(params)
    _ = model.fit(X, y)
    params.priors.value()[0] = 0.9
    assert_equal(model.class_prior_.item(0), 0.2)
    assert_equal(model.get_params().priors.value()[0], 0.2)
    var other: List[Int] = [1, 1, 1, 1]
    params.priors = None
    _ = model.set_params(params)
    _ = model.fit(X, other)
    assert_equal(model.class_count_.size, 1)
    assert_equal(model.classes_[0], 1)

    # Non-contiguous, negative strides and offset retain sample/feature order.
    var view = X.view_with_layout(Shape(4, 2), nm.NDArrayStrides(-2, -1), 7)
    var reversed_y: List[Int] = [-2, -2, 9, 9]
    _ = model.fit(view, reversed_y)
    assert_equal(model.score(view, reversed_y), 1.0)
    assert_equal(model.theta_.item(0, 0), 3.5)
    X[Item(3, 1)] = 99.0
    assert_equal(model.theta_.item(0, 0), 3.5)
    X[Item(3, 1)] = 4.0

    var strings = StringTarget(["z", "z", "a", "a"])
    var named = GaussianNB[StringDomain]()
    _ = named.partial_fit(X, strings, classes=StringTarget(["z", "absent", "a"]))
    assert_equal(named.classes_.get(0), "a")
    assert_equal(named.classes_.get(1), "absent")
    assert_equal(named.class_count_.item(1), 0.0)
    assert_equal(named.predict(X).get(0), "z")
    assert_equal(named.score(X, strings), 1.0)
    _ = named.partial_fit(X, strings)
    assert_equal(named.class_count_.item(2), 4.0)
    strings.set(0, "changed")
    assert_equal(named.classes_.get(2), "z")
    with assert_raises():
        _ = named.partial_fit(X, strings)
    var unknown = StringTarget(["unknown", "z", "a", "a"])
    assert_equal(named.score(X, unknown), 0.75)
    assert_equal(clone(named).classes_.rows(), 0)

    var numeric = GaussianNB[NumericDomain[DType.float64]]()
    var labels = nm.zeros[DType.float64](Shape(4))
    labels[Item(0)] = 9.0
    labels[Item(1)] = 9.0
    labels[Item(2)] = -2.0
    labels[Item(3)] = -2.0
    var X32 = X.astype[DType.float32]()
    _ = numeric.fit(X32, labels)
    assert_equal(numeric.classes_.item(0), -2.0)
    assert_equal(numeric.predict(X32).item(0), 9.0)
    assert_equal(numeric.score(X32, labels), 1.0)
    labels[Item(0)] = 0.5
    with assert_raises():
        _ = numeric.fit(X32, labels)
    labels[Item(0)] = Float64("inf")
    with assert_raises():
        _ = numeric.fit(X32, labels)
    with assert_raises():
        _ = numeric.fit(X32, nm.zeros[DType.float64](Shape(4, 1)))
    print("PASS: Gaussian arrays, labels, lifecycle, validation, strides and ownership")
