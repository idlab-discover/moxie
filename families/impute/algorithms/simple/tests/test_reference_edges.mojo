"""Reference edge cases checked against the pinned sklearn/impute/_base.py.

Expected values were produced by the pinned SimpleImputer/MissingIndicator
bodies (revision 4358542a1bd39af2e7e1cbace8b9fed067473c5b) on the same inputs.
All expected values are small integers or exact halves, so equality is exact.
"""
import numojo as nm
from std.testing import assert_equal, assert_raises, assert_true
from moxie.impute import SimpleImputer, MissingIndicator
from moxie.impute._plumbing import MissingValue


def matrix[dtype: DType](values: List[Float64], rows: Int, columns: Int) raises -> nm.NDArray[dtype]:
    var out = nm.zeros[dtype](nm.Shape(rows, columns))
    for i in range(len(values)):
        out.unsafe_set(i, Float64(values[i]).cast[dtype]())
    return out^


def check[dtype: DType]() raises:
    var nan = Float64("nan")
    var train = matrix[dtype]([1, 2, nan, 4], 2, 2)
    var test = matrix[dtype]([nan, nan, 1, 1], 2, 2)

    # features="all" never applies error_on_new: sklearn only checks new
    # missingness for "missing-only". Output keeps every original column.
    var every = MissingIndicator[dtype](features="all")
    _ = every.fit(train)
    var all_mask = every.transform(test)
    assert_equal(all_mask.shape[1], 2)
    assert_true(all_mask.unsafe_get(0) and all_mask.unsafe_get(1))
    assert_true(not all_mask.unsafe_get(2) and not all_mask.unsafe_get(3))

    # error_on_new=False keeps only the features missing during fit.
    var lenient = MissingIndicator[dtype](error_on_new=False)
    _ = lenient.fit(train)
    assert_equal(lenient.features_, [0])
    var lenient_mask = lenient.transform(test)
    assert_equal(lenient_mask.shape[1], 1)
    assert_true(lenient_mask.unsafe_get(0) and not lenient_mask.unsafe_get(1))
    var strict = MissingIndicator[dtype]()
    _ = strict.fit(train)
    with assert_raises(contains="new missing feature"):
        _ = strict.transform(test)

    # A finite marker makes NaN invalid input (ensure_all_finite=True).
    var with_nan = matrix[dtype]([1, nan, -1, 2], 2, 2)
    var finite_imputer = SimpleImputer[dtype](missing_values=MissingValue[dtype](-1))
    with assert_raises(contains="finite"):
        _ = finite_imputer.fit(with_nan)
    var finite_indicator = MissingIndicator[dtype](missing_values=MissingValue[dtype](-1))
    with assert_raises(contains="finite"):
        _ = finite_indicator.fit(with_nan)

    # Constant strategy drops fit-empty columns unless keep_empty_features:
    # statistics_ = [nan, 7], output keeps x1 only.
    var empty_first = matrix[dtype]([nan, 1, nan, nan], 2, 2)
    var constant = SimpleImputer[dtype](strategy="constant", fill_value=SIMD[dtype, 1](7))
    var filled = constant.fit_transform(empty_first)
    assert_true(constant.statistics_.unsafe_get(0) != constant.statistics_.unsafe_get(0))
    assert_equal(constant.statistics_.unsafe_get(1), 7)
    assert_equal(filled.shape[1], 1)
    assert_equal(Float64(filled.unsafe_get(0)), 1)
    assert_equal(Float64(filled.unsafe_get(1)), 7)
    assert_equal(constant.get_feature_names_out(), [String("x1")])

    # most_frequent with keep_empty_features fills an empty column with 0.
    var frequent = SimpleImputer[dtype](strategy="most_frequent", keep_empty_features=True)
    _ = frequent.fit(matrix[dtype]([nan, 1, nan, 1, nan, 2], 3, 2))
    assert_equal(frequent.statistics_.unsafe_get(0), 0)
    assert_equal(frequent.statistics_.unsafe_get(1), 1)
    var row = frequent.transform(matrix[dtype]([nan, nan], 1, 2))
    assert_equal(Float64(row.unsafe_get(0)), 0)
    assert_equal(Float64(row.unsafe_get(1)), 1)

    # Even-count median averages the middle pair: [1, 2, 4, 10] -> 3.
    var median = SimpleImputer[dtype](strategy="median")
    _ = median.fit(matrix[dtype]([1, 2, 4, 10, nan], 5, 1))
    assert_equal(median.statistics_.unsafe_get(0), 3)

    # Mean imputes a column that had no missing values in fit; the indicator
    # covers only fit-missing columns and does not raise on new missingness.
    var mean = SimpleImputer[dtype](add_indicator=True)
    _ = mean.fit(matrix[dtype]([1, nan, 3, 4], 2, 2))
    var out = mean.transform(matrix[dtype]([nan, nan], 1, 2))
    assert_equal(out.shape[1], 3)
    assert_equal(Float64(out.unsafe_get(0)), 2)
    assert_equal(Float64(out.unsafe_get(1)), 4)
    assert_equal(Float64(out.unsafe_get(2)), 1)
    assert_equal(mean.get_feature_names_out(), [String("x0"), String("x1"), String("missingindicator_x1")])


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: reference edges for indicators, markers, constant and empty features")
