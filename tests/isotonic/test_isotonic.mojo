"""Numerical, validation, ownership, and concrete estimator contract tests."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import isfinite
from moxie.base import clone, WeightedSupervisedEstimator
from moxie.isotonic import (
    IsotonicRegression,
    IsotonicRegressorAPI,
    Increasing,
    isotonic_regression,
    check_increasing,
)


def close(actual: List[Float64], expected: List[Float64]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_true(
            abs(actual[i] - expected[i]) <= 1e-12 * (1 + abs(expected[i]))
        )


def generic_fit[
    E: IsotonicRegressorAPI & WeightedSupervisedEstimator
](
    mut model: E,
    X: E.Input,
    y: E.Target,
    weights: E.SampleWeight,
) raises -> E.Prediction:
    _ = model.fit(X, y, sample_weight=weights)
    return model.predict(X)


def main() raises:
    var y: List[Float64] = [3, 7, 5, 9, 8, 7, 10]
    var X: List[Float64] = [0, 1, 2, 3, 4, 5, 6]
    close(isotonic_regression(y), [3, 6, 6, 8, 8, 8, 10])
    close(isotonic_regression([10.0, 0, 2]), [4.0, 4, 4])
    close(
        isotonic_regression([10.0, 9, 10, 7, 6, 6.1, 5], increasing=False),
        [10.0, 9.5, 9.5, 7, 6.05, 6.05, 5],
    )
    close(
        isotonic_regression([3.0, 0], sample_weight=List[Float64]([1.0, 2])),
        [1.0, 1],
    )
    close(isotonic_regression(y, y_min=4.0, y_max=7.0), [4.0, 6, 6, 7, 7, 7, 7])
    close(isotonic_regression(y, y_min=8.0, y_max=2.0), [2.0, 2, 2, 2, 2, 2, 2])
    close(y, [3.0, 7, 5, 9, 8, 7, 10])

    var model = IsotonicRegression()
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.f_(X)
    with assert_raises():
        _ = model.transform(X)
    _ = model.fit(X, y)
    close(model.predict([0.5, 1.5, 2.5, 4.5, 5.5]), [4.5, 6, 7, 8, 9])
    close(model.X_thresholds_, [0.0, 1, 2, 3, 5, 6])
    close(model.y_thresholds_, [3.0, 6, 6, 8, 8, 10])
    close(model.transform(X), model.predict(X))
    close(model.fit_transform(X, y.copy()), model.predict(X))
    assert_equal(model.X_min_, 0.0)
    assert_equal(model.X_max_, 6.0)
    assert_equal(model.get_feature_names_out()[0], "isotonicregression0")
    assert_true(model.increasing_)
    assert_false(isfinite(model.predict([-1.0])[0]))
    _ = model.set_params(out_of_bounds=String("clip"))
    close(model.predict([-1.0, 7]), [3.0, 10])
    assert_false(isfinite(model.f_([-1.0])[0]))
    # interp1d binds bounds_error at fit time, whereas input clipping is dynamic.
    _ = model.set_params(out_of_bounds=String("raise"))
    assert_false(isfinite(model.predict([-1.0])[0]))
    _ = model.fit(X, y)
    with assert_raises():
        _ = model.predict([-1.0])

    var weights: List[Float64] = [1, 2, 3, 4, 5, 6, 7]
    var targets: List[Float64] = [1, 41, 51, 1, 2, 5, 24]
    close(
        generic_fit(model, X, targets, weights),
        [1.0, 13.95, 13.95, 13.95, 13.95, 13.95, 24],
    )
    close(weights, [1.0, 2, 3, 4, 5, 6, 7])
    var shuffled = IsotonicRegression()
    _ = shuffled.fit(
        [6.0, 4, 2, 0, 1, 5, 3],
        [24.0, 2, 51, 1, 41, 5, 1],
        sample_weight=List[Float64]([7.0, 5, 3, 1, 2, 6, 4]),
    )
    close(shuffled.predict(X), model.predict(X))
    close(
        model.fit_transform(
            [0.0, 1, 1, 2, 3, 4], List[Float64]([0.0, 0, 1, 0, 0, 1])
        ),
        [0.0, 0.25, 0.25, 0.25, 0.25, 1],
    )
    _ = model.fit(
        [2.0, 0, 1, 1],
        [3.0, 0, 2, 6],
        sample_weight=List[Float64]([1.0, 1, 3, 1]),
    )
    close(model.predict([0.0, 0.5, 1, 1.5, 2]), [0.0, 1.5, 3, 3, 3])
    _ = model.fit([0.0, 1e-16, 1, 1 + 1e-14], [0.0, 0, 1, 2])
    close(model.X_thresholds_, [0.0, 1, 1 + 1e-14])
    _ = model.fit(
        [0.0, 1, 2, 3],
        [9.0, 1, 2, 8],
        sample_weight=List[Float64]([0.0, 1, 1, -1]),
    )
    assert_equal(model.X_min_, 1.0)
    assert_equal(model.X_max_, 2.0)
    for mode in [String("nan"), String("clip"), String("raise")]:
        var singleton = IsotonicRegression(out_of_bounds=mode.copy())
        _ = singleton.fit([1.0, 1], [2.0, 4])
        close(singleton.predict([-10.0, 1, 10]), [3.0, 3, 3])

    var automatic = IsotonicRegression(increasing=String("auto"))
    _ = automatic.fit([0.0, 1, 2], [3.0, 2, 1])
    assert_false(automatic.increasing_)
    _ = automatic.fit([0.0, 1, 2], [1.0, 2, 3])
    assert_true(automatic.increasing_)
    assert_true(check_increasing([0.0, 1, 2], [1.0, 1.1, 1.05]))
    assert_false(check_increasing([0.0, 1, 2], [1.0, 1, 1]))
    assert_false(check_increasing([1.0], [2.0]))
    assert_true(check_increasing([0.0, 1, 2, 3], [1.0, 3, 0, 2]))  # rho == 0

    var original = IsotonicRegression(
        y_min=0.0, increasing=False, out_of_bounds="clip"
    )
    _ = original.fit([0.0, 1], [3.0, 1])
    var copied = original.copy()
    var fresh = clone(original)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.y_min.value(), 0.0)
    assert_false(fresh.increasing[Bool])
    var params = original.get_params()
    params.y_min = None
    params.increasing = Increasing(True)
    _ = fresh.set_params(params)
    assert_false(Bool(fresh.y_min))
    assert_equal(original.y_min.value(), 0.0)
    copied.X_thresholds_[0] = -1.0
    assert_equal(original.X_thresholds_[0], 0.0)
    _ = original.fit([2.0], [9.0])
    assert_equal(len(original.X_thresholds_), 1)
    close(copied.predict([1.0]), [1.0])
    _ = fresh.set_params(y_min=2.0, increasing=Increasing(False))
    assert_equal(fresh.y_min.value(), 2.0)
    assert_false(fresh.increasing[Bool])

    var column: List[List[Float64]] = [
        List[Float64]([0.0]),
        List[Float64]([1.0]),
    ]
    var two_d = IsotonicRegression()
    close(two_d.fit_transform(column, [1.0, 2]), [1.0, 2])
    close(two_d.predict(column), [1.0, 2])
    close(two_d.transform(column), [1.0, 2])
    assert_equal(two_d.score(column, [1.0, 2]), 1.0)
    assert_equal(two_d.score([0.0, 1], [1.0, 2]), 1.0)
    assert_equal(two_d.score([0.0, 1], [1.0, 1]), 0.0)
    assert_false(isfinite(two_d.score([0.0], [1.0])))
    _ = two_d.fit([0.0, 1], [2.0, 2])
    assert_equal(two_d.score([0.0, 1], [2.0, 2]), 1.0)
    assert_true(
        abs(two_d.score([0.0, 1], [1.0, 4], List[Float64]([1.0, 2])) + 0.5)
        < 1e-12
    )

    with assert_raises():
        _ = isotonic_regression(List[Float64]())
    with assert_raises():
        _ = isotonic_regression([1.0], sample_weight=List[Float64]([0.0]))
    with assert_raises():
        _ = isotonic_regression([1.0], sample_weight=List[Float64]([-1.0]))
    with assert_raises():
        _ = isotonic_regression([1.0], sample_weight=List[Float64]([1.0, 1]))
    with assert_raises():
        _ = isotonic_regression([Float64("nan")])
    with assert_raises():
        _ = check_increasing([1.0], [1.0, 2])
    with assert_raises():
        _ = model.fit(X, [1.0])
    with assert_raises():
        _ = model.fit([1.0], [1.0], sample_weight=List[Float64]([0.0]))
    with assert_raises():
        _ = model.fit([1.0], [Float64("inf")])
    with assert_raises():
        _ = model.fit([Float64("nan")], [1.0])
    with assert_raises():
        _ = model.fit([1.0], [1.0], sample_weight=List[Float64]([Float64("nan")]))
    with assert_raises():
        _ = model.predict(List[Float64]())
    with assert_raises():
        _ = model.predict([Float64("inf")])
    with assert_raises():
        _ = model.fit_transform(X)
    with assert_raises():
        _ = model.fit(List[List[Float64]]([List[Float64]([0.0, 1])]), [1.0])
    with assert_raises():
        _ = model.predict(List[List[Float64]]([List[Float64]([0.0, 1])]))
    with assert_raises():
        _ = two_d.score([0.0, 1], [1.0, 2], List[Float64]([0.0, 0]))
    with assert_raises():
        var invalid = IsotonicRegression(out_of_bounds="other")
        _ = invalid.fit(X, y)
    with assert_raises():
        var invalid = IsotonicRegression(increasing="other")
        _ = invalid.fit(X, y)
    with assert_raises():
        _ = isotonic_regression([1.0], y_min=Float64("nan"))
    with assert_raises():
        var invalid = IsotonicRegression(y_max=Float64("nan"))
        _ = invalid.fit(X, y)
    with assert_raises():
        _ = check_increasing(List[Float64](), List[Float64]())
    with assert_raises():
        _ = check_increasing([Float64("nan")], [1.0])
    close(
        isotonic_regression(
            [1.0, 2], y_min=Float64("-inf"), y_max=Float64("inf")
        ),
        [1.0, 2],
    )
    # Failed refits leave the previous valid fitted state intact.
    assert_equal(model.X_min_, 1.0)
    assert_equal(model.X_max_, 2.0)
    print("Isotonic numerical and estimator contract checks passed")
