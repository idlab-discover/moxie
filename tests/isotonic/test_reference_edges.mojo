"""Edge values from the pinned sklearn/isotonic.py (4358542a), derived by hand.

Each block names the reference behavior it pins; expected values follow from
the reference algorithm (lexsort, _make_unique, PAVA, trimming, r2_score).
"""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import isnan
from moxie.base import clone
from moxie.isotonic import IsotonicRegression, Increasing, check_increasing


def close(actual: List[Float64], expected: List[Float64]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_true(abs(actual[i] - expected[i]) <= 1e-12 * (1 + abs(expected[i])))


def main() raises:
    # A violating pair pools to its mean; the interior 0.5 is kept because its
    # right neighbor differs (keep_data in _build_y).
    var model = IsotonicRegression()
    _ = model.fit([0.0, 1, 2], [1.0, 0, 3])
    close(model.X_thresholds_, [0.0, 1, 2])
    close(model.y_thresholds_, [0.5, 0.5, 3])
    # Predictions at thresholds reproduce y_thresholds_, between them interpolate.
    close(model.predict([0.0, 1, 2, 1.5]), [0.5, 0.5, 3, 1.75])

    # Determinism: refitting the same data gives bitwise-identical state.
    var first = model.y_thresholds_.copy()
    _ = model.fit([0.0, 1, 2], [1.0, 0, 3])
    for i in range(len(first)):
        assert_equal(first[i], model.y_thresholds_[i])

    # y_min > y_max clips everything to y_max (np.clip); trimming keeps both ends.
    var crossed = IsotonicRegression(y_min=3.0, y_max=1.0)
    _ = crossed.fit([0.0, 1, 2], [1.0, 0, 3])
    close(crossed.X_thresholds_, [0.0, 2])
    close(crossed.y_thresholds_, [1.0, 1])

    # r2_score accepts negative weights: mean 2.5, SSE -6.5, SST 1.5 -> 1 + 6.5/1.5.
    assert_true(
        abs(model.score([0.0, 1], [1.0, 2], List[Float64]([1.0, -3])) - 16.0 / 3.0)
        < 1e-12
    )

    # Spearman with tied ranks: rho = 0.447 > 0, so "auto" picks increasing.
    var automatic = IsotonicRegression(increasing="auto")
    _ = automatic.fit([0.0, 1, 2, 3], [0.0, 1, 0, 1])
    assert_true(automatic.increasing_)
    close(automatic.y_thresholds_, [0.0, 0.5, 0.5, 1])

    # set_params then refit replaces the learned state; clone keeps the new
    # configuration but no fitted buffers.
    _ = model.set_params(increasing=Increasing(False), y_max=0.75)
    _ = model.fit([0.0, 1, 2], [1.0, 0, 3])
    assert_false(model.increasing_)
    close(model.X_thresholds_, [0.0, 2])
    close(model.y_thresholds_, [0.75, 0.75])
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.increasing[Bool])
    assert_equal(fresh.y_max.value(), 0.75)
    assert_equal(len(fresh.X_thresholds_), 0)

    # Every sample filtered by nonpositive weight: the reference raises from
    # check_array on the empty remainder; Moxie raises and keeps the old state.
    with assert_raises():
        _ = model.fit([0.0, 1], [1.0, 2], sample_weight=List[Float64]([-1.0, 0]))
    close(model.y_thresholds_, [0.75, 0.75])

    # Float32 "nan" mode yields NaN outside the domain, same dtype.
    var single = IsotonicRegression[DType.float32]()
    var x32: List[Float32] = [0, 1]
    var y32: List[Float32] = [1, 2]
    var query32: List[Float32] = [-1, 2]
    _ = single.fit(x32, y32)
    var outside = single.predict(query32)
    assert_true(isnan(outside[0]) and isnan(outside[1]))

    # Documented validation difference: the reference returns True for an
    # infinite y (spearmanr ranks it last); Moxie rejects nonfinite inputs.
    with assert_raises(contains="finite"):
        _ = check_increasing([0.0, 1, 2], [1.0, 2, Float64("inf")])
    print("Isotonic reference edge checks passed")
