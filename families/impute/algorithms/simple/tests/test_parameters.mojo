import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie import clone
from moxie.impute import SimpleImputer, MissingIndicator
from moxie.impute._plumbing import MissingValue
from moxie.impute._simple import ColumnStrategy


struct FixedStatistic(ColumnStrategy):
    var value: Float64

    def __init__(out self):
        self.value = 3

    def __init__(out self, value: Float64):
        self.value = value

    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        return self.value


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(2, 1))
    X.unsafe_set(0, -1)
    X.unsafe_set(1, 5)
    var model = SimpleImputer[dtype](strategy="constant", fill_value=SIMD[dtype, 1](9))
    _ = model.fit(X)
    _ = model.set_params(missing_values=MissingValue[dtype](-1), copy=False, add_indicator=True)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.transform(X)
    assert_equal(model.get_params().fill_value.value(), SIMD[dtype, 1](9))
    var filled = model.fit_transform(X)
    assert_equal(filled.unsafe_get(0), 9)
    assert_equal(filled.shape[1], 2)
    _ = model.set_params(fill_value=SIMD[dtype, 1](7), add_indicator=False)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(model.fit_transform(X).unsafe_get(0), 7)
    _ = model.set_params(fill_value=None)
    assert_false(model.get_params().fill_value)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(model.fit_transform(X).unsafe_get(0), 0)
    _ = model.set_params(missing_values=MissingValue[dtype](), strategy="mean")
    X.unsafe_set(0, Float64("nan").cast[dtype]())
    assert_equal(model.fit_transform(X).unsafe_get(0), 5)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().strategy, "mean")

    X.unsafe_set(0, -1)
    var indicator = MissingIndicator[dtype]()
    _ = indicator.fit(X)
    _ = indicator.set_params(missing_values=MissingValue[dtype](-1), features="all", sparse=False)
    assert_false(indicator.__sklearn_is_fitted__())
    assert_equal(indicator.fit_transform(X).unsafe_get(0), True)
    _ = indicator.set_params(missing_values=MissingValue[dtype](5), sparse="auto", error_on_new=False)
    assert_false(indicator.__sklearn_is_fitted__())
    var mask = indicator.fit_transform(X)
    assert_equal(mask.unsafe_get(0), False)
    assert_equal(mask.unsafe_get(1), True)
    var indicator_clone = clone(indicator)
    assert_equal(indicator_clone.get_params().missing_values.value, SIMD[dtype, 1](5))
    assert_equal(indicator_clone.get_params().sparse[String], "auto")

    var custom = SimpleImputer[dtype, FixedStatistic](strategy="callable", missing_values=MissingValue[dtype](-1))
    assert_equal(custom.fit_transform(X).unsafe_get(0), 3)
    _ = custom.set_params(callback=FixedStatistic(12))
    assert_false(custom.__sklearn_is_fitted__())
    assert_equal(custom.fit_transform(X).unsafe_get(0), 12)
    var custom_clone = clone(custom)
    assert_false(custom_clone.__sklearn_is_fitted__())
    assert_equal(custom_clone.fit_transform(X).unsafe_get(0), 12)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: keyword parameter updates, fill reset and callback cloning")
