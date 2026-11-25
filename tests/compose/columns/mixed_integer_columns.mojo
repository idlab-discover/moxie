"""Int64 features in a mixed numeric pipeline with a categorical branch."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import NDArray, NDArrayStrides, Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises


def table(
    counts: List[Int64], ages: List[Float64], colors: List[Optional[String]],
    count_first: Bool,
) raises -> ColumnTable[]:
    var age = zeros[DType.float64](Shape(len(ages), 1))
    var backing = zeros[DType.int64](Shape(2 * len(counts)))
    for row in range(len(ages)):
        age.unsafe_set(row, ages[row])
        backing.unsafe_set(backing.size - 1 - 2 * row, counts[row])
    var count = backing.view_with_layout(
        Shape(len(counts)), NDArrayStrides(-2), backing.size - 1,
    )
    var result = ColumnTable(age, ["age"])
    result.add_categorical("color", categorical_column(colors))
    result.add_numeric("count", count, position=0 if count_first else 2)
    backing.unsafe_set(backing.size - 1, 999)
    return result^


def main() raises:
    var train = table(
        [-4, 0, 7, 0, 13], [10.0, nan[DType.float64](), 30.0, 40.0, 20.0],
        ["red", "blue", "red", "amber", "blue"], True,
    )
    var test = table(
        [0, -4, 9007199254740993], [35.0, nan[DType.float64](), 15.0],
        ["green", None, "red"], False,
    )
    var numeric_kinds: List[String] = ["number"]
    var category_kinds: List[String] = ["category"]
    var number = make_column_selector(dtype_include=numeric_kinds.copy())
    var category = make_column_selector(dtype_include=category_kinds^)
    assert_equal(train.schema().dtypes, ["int64", "float64", "category"])
    assert_equal(number(train).names, ["count", "age"])
    for kind in ["int", "int64", "integer", "signedinteger"]:
        var kinds: List[String] = [kind]
        assert_equal(make_column_selector(dtype_include=kinds^)(train).names, ["count"])
    var floats: List[String] = ["floating"]
    var unsigned: List[String] = ["unsignedinteger"]
    var integers: List[String] = ["integer"]
    assert_equal(make_column_selector(dtype_include=floats^)(train).names, ["age"])
    assert_equal(make_column_selector(dtype_include=unsigned^)(train).names, List[String]())
    assert_equal(
        make_column_selector(dtype_include=numeric_kinds^, dtype_exclude=integers^)(train).names,
        ["age"],
    )
    with assert_raises(contains="rank-1"):
        train.add_numeric("invalid", zeros[DType.int64](Shape(5, 1)))
    with assert_raises(contains="same number of rows"):
        train.add_numeric("invalid", zeros[DType.int64](Shape(3)))
    assert_equal(train.schema().names, ["count", "age", "color"])

    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), number^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), category^),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)
    var result = ResultLine("mixed_integer_columns")
    var fitted: NDArray[DType.float64] = model.fit_transform(train.copy())
    result.add("fit", fitted)
    result.add("transform", model.transform(test))
    var float_test = ColumnTable(zeros[DType.float64](Shape(3, 0)), [])
    var count = zeros[DType.float64](Shape(3))
    count.unsafe_set(1, -4)
    # The Int64 test value rounds to 2**53 at the Float64 branch boundary.
    count.unsafe_set(2, 9007199254740992.0)
    var age = zeros[DType.float64](Shape(3))
    age.unsafe_set(0, 35)
    age.unsafe_set(1, nan[DType.float64]())
    age.unsafe_set(2, 15)
    float_test.add_numeric("count", count)
    float_test.add_numeric("age", age)
    float_test.add_categorical("color", test.categorical_columns([1])[0])
    result.add("after_dtype_change", model.transform(float_test))
    var invalid = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    invalid.add_numeric("count", zeros[DType.int64](Shape(3)))
    invalid.add_strings("color", ["green", "", "red"])
    with assert_raises(contains="requires category storage"):
        _ = model.transform(invalid)
    # Keep color selected, but provide a category domain the encoder cannot read.
    var wrong = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    wrong.add_numeric("count", zeros[DType.int64](Shape(3)))
    wrong.add_categorical("color", categorical_column(zeros[DType.int64](Shape(3))))
    with assert_raises(contains="different categorical domain"):
        _ = model.fit_transform(wrong)
    result.add("after_failed_refit", model.transform(test))
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
