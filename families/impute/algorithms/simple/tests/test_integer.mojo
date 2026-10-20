import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie import clone
from moxie.impute import IntegerSimpleImputer
from moxie.impute._plumbing import MissingValue
from moxie.impute._simple import ColumnStrategy

struct Minimum(ColumnStrategy):
    def __init__(out self):
        pass
    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        if values.size == 0:
            return 17
        return nm.min(values)


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(3, 3))
    var data: List[Int] = [1, 0, 0, 4, 2, 0, 0, 7, 0]
    for i in range(9):
        X.unsafe_set(i, SIMD[dtype, 1](data[i]))
    for strategy in [String("mean"), String("median"), String("most_frequent"), String("constant")]:
        var model = IntegerSimpleImputer[dtype](strategy=strategy, missing_values=MissingValue[dtype](0), add_indicator=True)
        with assert_raises():
            _ = model.transform(X)
        var output = model.fit_transform(X)
        if strategy == "mean" or strategy == "median":
            assert_true(output.isa[nm.NDArray[DType.float64]]())
            assert_equal(output[nm.NDArray[DType.float64]].shape[1], 5)
            assert_equal(output[nm.NDArray[DType.float64]].unsafe_get(1), 4.5)
            assert_equal(output[nm.NDArray[DType.float64]].unsafe_get(10), 2.5)
        else:
            assert_true(output.isa[nm.NDArray[dtype]]())
            assert_equal(output[nm.NDArray[dtype]].shape[1], 5)
            assert_equal(output[nm.NDArray[dtype]].unsafe_get(1), SIMD[dtype, 1](2 if strategy == "most_frequent" else 0))
            assert_equal(output[nm.NDArray[dtype]].unsafe_get(4), 1)
        var inverse = model.inverse_transform(output)
        for i in range(9):
            assert_equal(inverse.unsafe_get(i), Float64(X.unsafe_get(i)))
        assert_true(model.statistics_.unsafe_get(2) != model.statistics_.unsafe_get(2))
        var names = model.get_feature_names_out()
        assert_equal(len(names), 5)
        assert_equal(names[2], "missingindicator_x0")
        var copied = clone(model)
        assert_false(copied.__sklearn_is_fitted__())
        _ = model.set_params(strategy="constant", fill_value=SIMD[dtype, 1](9), keep_empty_features=True)
        assert_false(model.__sklearn_is_fitted__())
        var kept = model.fit_transform(X)
        assert_equal(kept[nm.NDArray[dtype]].shape[1], 6)
        assert_equal(kept[nm.NDArray[dtype]].unsafe_get(2), 9)
        _ = model.set_params(fill_value=None)
        var cleared = model.fit_transform(X)
        assert_equal(cleared[nm.NDArray[dtype]].unsafe_get(2), 0)
        var empty = nm.zeros[dtype](nm.Shape(0, 3))
        with assert_raises():
            _ = model.fit(empty)
        assert_true(model.__sklearn_is_fitted__())
        with assert_raises():
            _ = model.transform(nm.zeros[dtype](nm.Shape(2, 4)))
    var custom = IntegerSimpleImputer[dtype, Minimum](strategy="callable", missing_values=MissingValue[dtype](0))
    var result = custom.fit_transform(X)
    assert_equal(result[nm.NDArray[DType.float64]].unsafe_get(2), 17)
    var complete = IntegerSimpleImputer[dtype](strategy="most_frequent")
    var unchanged = complete.fit_transform(X)
    assert_equal(unchanged[nm.NDArray[dtype]].shape[1], 3)
    assert_equal(unchanged[nm.NDArray[dtype]].unsafe_get(1), 0)
    unchanged[nm.NDArray[dtype]].unsafe_set(0, 99)
    assert_equal(X.unsafe_get(0), 1)


def strided[dtype: DType]() raises:
    var backing = nm.zeros[dtype](nm.Shape(3, 3))
    for i in range(9):
        backing.unsafe_set(i, SIMD[dtype, 1](i))
    backing.unsafe_set(6, 0)
    var view = backing.view_with_layout(nm.Shape(2, 2), nm.Strides(-3, 1), 6)
    var contiguous = nm.zeros[dtype](nm.Shape(2, 2))
    for i in range(2):
        for j in range(2):
            contiguous.unsafe_set(i * 2 + j, view.unsafe_get(-i * 3 + j))
    for strategy in [String("mean"), String("most_frequent"), String("constant")]:
        var first = IntegerSimpleImputer[dtype](strategy=strategy, missing_values=MissingValue[dtype](0), add_indicator=True)
        var second = IntegerSimpleImputer[dtype](strategy=strategy, missing_values=MissingValue[dtype](0), add_indicator=True)
        var a = first.fit_transform(view)
        var b = second.fit_transform(contiguous)
        if strategy == "mean":
            for i in range(a[nm.NDArray[DType.float64]].size):
                assert_equal(a[nm.NDArray[DType.float64]].unsafe_get(i), b[nm.NDArray[DType.float64]].unsafe_get(i))
        else:
            for i in range(a[nm.NDArray[dtype]].size):
                assert_equal(a[nm.NDArray[dtype]].unsafe_get(i), b[nm.NDArray[dtype]].unsafe_get(i))
        assert_equal(backing.unsafe_get(6), 0)


def large_values() raises:
    var X = nm.zeros[DType.uint64](nm.Shape(2, 1))
    X.unsafe_set(0, UInt64(18446744073709551615))
    var model = IntegerSimpleImputer[DType.uint64](strategy="constant", fill_value=UInt64(18446744073709551615), missing_values=MissingValue[DType.uint64](0))
    var output = model.fit_transform(X)
    assert_equal(output[nm.NDArray[DType.uint64]].unsafe_get(1), UInt64(18446744073709551615))
    var signed = nm.zeros[DType.int64](nm.Shape(2, 1))
    signed.unsafe_set(0, Int64(9223372036854775807))
    var signed_model = IntegerSimpleImputer[DType.int64](strategy="most_frequent", missing_values=MissingValue[DType.int64](0))
    _ = signed_model.fit(signed)
    with assert_raises():
        _ = signed_model.transform(signed)
    _ = model.set_params(strategy="most_frequent")
    _ = model.fit(X)
    with assert_raises():
        _ = model.transform(X)


def main() raises:
    check[DType.int8]()
    check[DType.int16]()
    check[DType.int32]()
    check[DType.int64]()
    check[DType.uint8]()
    check[DType.uint16]()
    check[DType.uint32]()
    check[DType.uint64]()
    strided[DType.int64]()
    strided[DType.uint32]()
    large_values()
    print("PASS: integer simple strategies, dtypes, inverse, clone and updates")
