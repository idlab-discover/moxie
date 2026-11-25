"""Float32 and Float64 storage in one real mixed ColumnTransformer."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import NDArray, NDArrayStrides, Shape, zeros
from parity_io import ResultLine
from std.math import inf, nan
from std.testing import assert_equal, assert_false, assert_raises


def table(
    ages: List[Float64], ratings: List[Float32], colors: List[Optional[String]],
    rating_first: Bool,
) raises -> ColumnTable[]:
    var age = zeros[DType.float64](Shape(len(ages), 1))
    var backing = zeros[DType.float32](Shape(2 * len(ratings)))
    for row in range(len(ages)):
        age.unsafe_set(row, ages[row])
        backing.unsafe_set(2 * (len(ratings) - row - 1) + 1, ratings[row])
    var rating = backing.view_with_layout(Shape(len(ratings)), NDArrayStrides(-2), backing.size - 1)
    var result = ColumnTable(age, ["age"])
    result.add_categorical("color", categorical_column(colors), position=1 if rating_first else 0)
    result.add_numeric("rating", rating, position=0 if rating_first else 2)
    backing.unsafe_set(backing.size - 1, 999)
    age.unsafe_set(0, 999)
    return result^


def main() raises:
    var train = table(
        [10.0, nan[DType.float64](), 30.0, 40.0, 20.0],
        [1.25, 2.5, nan[DType.float32](), 1.25, 5.0],
        ["red", "blue", "red", "amber", "blue"], True,
    )
    var test = table(
        [35.0, nan[DType.float64](), 15.0], [1.25, nan[DType.float32](), 2.5],
        ["green", None, "red"], False,
    )
    var number_kinds: List[String] = ["number"]
    var category_kinds: List[String] = ["category"]
    var float32_kinds: List[String] = ["float32"]
    var floating_kinds: List[String] = ["floating"]
    var number = make_column_selector(dtype_include=number_kinds^)
    var category = make_column_selector(dtype_include=category_kinds^)
    assert_equal(train.schema().dtypes, ["float32", "float64", "category"])
    assert_equal(number(train).names, ["rating", "age"])
    assert_equal(category(train).names, ["color"])
    assert_equal(make_column_selector(dtype_include=float32_kinds.copy())(train).names, ["rating"])
    assert_equal(
        make_column_selector(dtype_include=floating_kinds^, dtype_exclude=float32_kinds^)(train).names,
        ["age"],
    )
    with assert_raises(contains="rank-1"):
        train.add_numeric("invalid", zeros[DType.float32](Shape(5, 1)))
    with assert_raises(contains="same number of rows"):
        train.add_numeric("invalid", zeros[DType.float32](Shape(3)))
    with assert_raises(contains="unique"):
        train.add_numeric("age", zeros[DType.float32](Shape(5)))
    with assert_raises(contains="out of bounds"):
        train.add_numeric("invalid", zeros[DType.float32](Shape(5)), position=4)
    assert_equal(train.schema().names, ["rating", "age", "color"])
    var empty = ColumnTable(zeros[DType.float64](Shape(0, 0)), [])
    var empty_backing = zeros[DType.float32](Shape(1))
    var empty_column = empty_backing.view_with_layout(Shape(0), NDArrayStrides(-2), 0)
    empty.add_numeric("rating", empty_column)
    assert_equal(empty.schema().dtypes, ["float32"])

    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), number^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), category^),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)
    var result = ResultLine("mixed_float_columns")
    var fitted: NDArray[DType.float64] = model.fit_transform(train)
    result.add("fit", fitted)
    result.add("transform", model.transform(test))
    var wide_test = ColumnTable(zeros[DType.float64](Shape(3, 0)), [])
    var wide_rating = zeros[DType.float64](Shape(3))
    wide_rating.unsafe_set(0, 1.25)
    wide_rating.unsafe_set(1, nan[DType.float64]())
    wide_rating.unsafe_set(2, 2.5)
    var wide_age = zeros[DType.float64](Shape(3))
    wide_age.unsafe_set(0, 35.0)
    wide_age.unsafe_set(1, nan[DType.float64]())
    wide_age.unsafe_set(2, 15.0)
    wide_test.add_numeric("rating", wide_rating)
    wide_test.add_numeric("age", wide_age)
    wide_test.add_categorical("color", test.categorical_columns([0])[0])
    result.add("after_dtype_change", model.transform(wide_test))
    var wrong = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    var colors: List[Optional[String]] = ["red", "red", "blue"]
    wrong.add_categorical("color", categorical_column(colors))
    wrong.add_strings("rating", ["fast", "slow", "fast"])
    with assert_raises(contains="requires numeric storage"):
        _ = model.transform(wrong)
    var invalid = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    wide_rating.unsafe_set(0, inf[DType.float64]())
    invalid.add_numeric("rating", wide_rating)
    invalid.add_categorical("color", categorical_column(colors))
    with assert_raises(contains="non-missing values must be finite"):
        _ = model.fit_transform(invalid)
    result.add("after_failed_refit", model.transform(test.copy()))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.fit(train)
    result.add("after_fit", fresh.transform(test))
    result.emit()
    for name in model.get_feature_names_out():
        print(String("name|", name))
    ref spans = model.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(String("span|", name, "|", span.start, "|", span.stop))
