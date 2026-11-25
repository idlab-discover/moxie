import numojo as nm
from std.math import abs, isnan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie.impute import IterativeBound, IterativeImputer
from moxie.impute._plumbing import MissingValue
from moxie.linear_model import BayesianRidge


def check_integer_input() raises:
    var X = nm.zeros[DType.int32](nm.Shape(4, 2))
    for i in range(4):
        X.unsafe_set(2 * i, Int32(i + 1))
        X.unsafe_set(2 * i + 1, Int32(2 * (i + 1)))
    X.unsafe_set(3, -1)
    var reversed = X.view_with_layout(nm.Shape(4, 2), nm.Strides(-2, 1), 6)
    var model = IterativeImputer(
        missing_values=MissingValue[DType.float64](-1), add_indicator=True, random_state=42,
    )
    with assert_raises():
        _ = model.transform(X)
    var filled: nm.NDArray[DType.float64] = model.fit(reversed).transform(X)
    assert_equal(filled.shape[1], 3)
    assert_true(abs(filled.unsafe_get(4) - 4) < 1e-5)
    assert_equal(filled.unsafe_get(5), 1)
    assert_equal(X.unsafe_get(3), Int32(-1))
    var names: List[String] = ["distance", "duration"]
    model.fit_named(reversed, names)
    var expected_names: List[String] = ["distance", "duration", "missingindicator_duration"]
    assert_equal(model.get_feature_names_out(), expected_names)
    with assert_raises():
        _ = model.fit(nm.zeros[DType.int32](nm.Shape(8)))
    with assert_raises():
        _ = model.fit(nm.zeros[DType.int32](nm.Shape(0, 2)))
    with assert_raises():
        _ = model.transform(nm.zeros[DType.int32](nm.Shape(4, 3)))
    assert_equal(model.get_feature_names_out(), expected_names)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var complete = nm.ones[DType.uint16](nm.Shape(4, 2))
    assert_equal(fresh.fit_transform(complete).shape[1], 2)
    var default_names: List[String] = ["x0", "x1"]
    assert_equal(fresh.get_feature_names_out(), default_names)


def check_posterior() raises:
    var X = nm.zeros[DType.float64](nm.Shape(5, 2))
    for i in range(5):
        X.unsafe_set(2 * i, Float64(i))
        X.unsafe_set(2 * i + 1, Float64(i % 2))
    X.unsafe_set(3, Float64("nan"))
    var model = IterativeImputer(
        sample_posterior=True, random_state=42, max_iter=3, tol=1e6,
    )
    with assert_raises():
        _ = model.transform(X)
    var fitted = model.fit_transform(X)
    assert_equal(model.n_iter_, 3)
    assert_equal(len(model.imputation_sequence_), 6)
    var first = model.transform(X)
    assert_true(fitted.unsafe_get(3) != first.unsafe_get(3))
    var copied = model.copy()
    var second = model.transform(X)
    assert_true(first.unsafe_get(3) != second.unsafe_get(3))
    assert_equal(second.unsafe_get(3), copied.transform(X).unsafe_get(3))
    with assert_raises():
        _ = model.transform(nm.zeros[DType.float64](nm.Shape(4, 3)))
    assert_equal(model.transform(X).unsafe_get(3), copied.transform(X).unsafe_get(3))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.fit_transform(X).unsafe_get(3), fitted.unsafe_get(3))
    assert_equal(fresh.transform(X).unsafe_get(3), first.unsafe_get(3))


def main() raises:
    check_integer_input()
    check_posterior()
    var missing = Float64("nan")
    var X = nm.zeros[DType.float64](nm.Shape(4, 2))
    for i in range(4):
        X.unsafe_set(2 * i, Float64(i + 1))
        X.unsafe_set(2 * i + 1, Float64(2 * (i + 1)))
    X.unsafe_set(3, missing)
    var reversed = X.view_with_layout(nm.Shape(4, 2), nm.Strides(-2, 1), 6)

    var base = BayesianRidge(max_iter=80)
    var y = nm.ones[DType.float64](nm.Shape(4))
    _ = base.fit(nm.ones[DType.float64](nm.Shape(4, 2)), y)
    var model = IterativeImputer(base.copy(), add_indicator=True, random_state=42)
    var params = model.get_params()
    assert_true(base.__sklearn_is_fitted__())
    assert_false(params.estimator.value().__sklearn_is_fitted__())
    assert_equal(params.estimator.value().max_iter, 80)
    with assert_raises():
        _ = model.transform(X)
    with assert_raises():
        _ = model.get_feature_names_out()

    var names: List[String] = ["distance", "duration"]
    model.fit_named(reversed, names)
    var expected_names: List[String] = [
        "distance", "duration", "missingindicator_duration",
    ]
    assert_equal(model.get_feature_names_out(), expected_names)
    assert_false(Bool(model.initial_imputer_.indicator_))
    assert_equal(model.indicator_.value().features_, List[Int](length=1, fill=1))
    assert_true(Bool(model.random_state_))
    with assert_raises():
        var wrong_names: List[String] = ["distance", "time"]
        _ = model.get_feature_names_out(wrong_names^)
    var result = model.transform(reversed)
    assert_equal(result.shape[0], 4)
    assert_equal(result.shape[1], 3)
    assert_equal(result.unsafe_get(6), 2)
    assert_true(abs(result.unsafe_get(7) - 4) < 1e-5)
    assert_equal(result.unsafe_get(8), 1)
    assert_true(isnan(X.unsafe_get(3)))

    # A failed fit preserves the last committed sequence and fitted metadata.
    with assert_raises():
        _ = model.fit(nm.zeros[DType.float64](nm.Shape(0, 2)))
    with assert_raises():
        _ = model.transform(nm.zeros[DType.float64](nm.Shape(4, 3)))
    assert_equal(model.get_feature_names_out(), expected_names)
    var copied = model.copy()
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(Bool(fresh.random_state_))
    assert_equal(len(fresh.imputation_sequence_), 0)
    assert_equal(fresh.get_params().random_state.value(), 42)
    _ = model.set_params(max_iter=0).fit(X)
    assert_equal(model.n_iter_, 0)
    assert_equal(len(model.imputation_sequence_), 0)
    assert_equal(model.transform(X).unsafe_get(4), 16.0 / 3.0)
    assert_true(abs(copied.transform(X).unsafe_get(4) - 4) < 1e-5)

    var wrong_bounds = IterativeImputer(
        min_value=IterativeBound(List[Float64](length=1, fill=0.0)),
    )
    with assert_raises():
        _ = wrong_bounds.fit(X)
    var inverted_bounds = IterativeImputer(
        min_value=IterativeBound(Float64(1)), max_value=IterativeBound(Float64(1)),
    )
    with assert_raises():
        _ = inverted_bounds.fit(X)
    var invalid_order = IterativeImputer(imputation_order="unknown")
    with assert_raises():
        _ = invalid_order.fit(X)
    var invalid_strategy = IterativeImputer(initial_strategy="callable")
    with assert_raises():
        _ = invalid_strategy.fit(X)
    var invalid_neighbors = IterativeImputer(n_nearest_features=0)
    with assert_raises():
        _ = invalid_neighbors.fit(X)
    var invalid_iters = IterativeImputer(max_iter=-1)
    with assert_raises():
        _ = invalid_iters.fit(X)
    var invalid_seed = IterativeImputer(random_state=-1)
    with assert_raises():
        _ = invalid_seed.fit(X)
    print("PASS: iterative imputer lifecycle, floating and integer layouts, and named metadata")
