import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.impute import SimpleImputer, MissingIndicator
from moxie.impute._plumbing import MissingValue
from moxie.impute._simple import ColumnStrategy

struct NanStatistic(ColumnStrategy):
    def __init__(out self):
        pass
    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        return Float64("nan")


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(3, 3))
    for i in range(9):
        X.unsafe_set(i, Float64(i))
    X.unsafe_set(0, Float64("nan"))
    var model = SimpleImputer(copy=False)
    var names: List[String] = ["a", "b", "c"]
    model.fit_named(X, names)
    assert_equal(model.get_feature_names_out(), names)
    names[0] = "changed"
    assert_equal(model.get_feature_names_out()[0], "a")
    with assert_raises():
        _ = model.get_feature_names_out(names.copy())
    var Y = model.transform(X)
    assert_true(X.unsafe_get(0) != X.unsafe_get(0))
    Y.unsafe_set(1, 99)
    assert_equal(X.unsafe_get(1), 1)
    # Fitted statistics own storage independent from later training mutation.
    X.unsafe_set(3, 100)
    assert_equal(model.statistics_.unsafe_get(0), 4.5)
    var small = nm.zeros[DType.float32](nm.Shape(1, 3))
    small.unsafe_set(0, Float64("nan").cast[DType.float32]())
    var transformed = model.transform[DType.float32](small)
    assert_equal(transformed.unsafe_get(0), Float32(4.5))
    _ = model.set_params(strategy=String("median"))
    assert_false(model.__sklearn_is_fitted__())
    _ = model.fit(X)
    assert_equal(model.get_feature_names_out()[0], "x0")
    # Arbitrary positive and negative element strides with offset.
    var strided = X.view_with_layout(nm.Shape(2, 2), nm.Strides(-3, 1), 6)
    var contiguous = nm.zeros[DType.float64](nm.Shape(2, 2))
    for i in range(2):
        for j in range(2):
            contiguous.unsafe_set(i * 2 + j, strided.unsafe_get(-i * 3 + j))
    var first = SimpleImputer()
    var second = SimpleImputer()
    var a = first.fit_transform(strided)
    var b = second.fit_transform(contiguous)
    for i in range(4):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))
    var all_empty = nm.full[DType.float64](nm.Shape(2, 2), Float64("nan"))
    var none = model.fit_transform(all_empty)
    assert_equal(none.shape[1], 0)
    var custom = SimpleImputer[DType.float64, NanStatistic](strategy="callable", callback=NanStatistic())
    assert_equal(custom.fit_transform(contiguous).shape[1], 0)
    _ = custom.set_params(keep_empty_features=True)
    assert_equal(custom.fit_transform(all_empty).shape[1], 2)
    var wrong = nm.zeros[DType.float64](nm.Shape(1))
    with assert_raises():
        _ = model.fit(wrong)
    var infinite = nm.full[DType.float64](nm.Shape(1, 2), Float64("inf"))
    with assert_raises():
        _ = model.fit(infinite)
    var invalid = SimpleImputer(strategy="invalid")
    with assert_raises():
        _ = invalid.fit(X)
    var sentinel = SimpleImputer(missing_values=MissingValue[DType.float64](-1))
    with assert_raises():
        _ = sentinel.fit(X)
    var indicator = MissingIndicator()
    indicator.fit_named(X, [String("a"), String("b"), String("c")])
    assert_equal(indicator.get_feature_names_out(), [String("missingindicator_a")])
    _ = indicator.set_params(features=String("all"))
    assert_false(indicator.__sklearn_is_fitted__())
    print("PASS: lifecycle, names, cross-dtype, stride and invalid inputs")
