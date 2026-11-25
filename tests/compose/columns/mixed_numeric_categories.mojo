"""Numeric category domains in a mixed table with native preprocessing."""
from moxie._categorical import NumericCategories, categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from moxie.preprocessing._vocabulary import NumericKind
from numojo import Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises

comptime ProductEncoder = OneHotEncoder[NumericKind[DType.int64]]
comptime LARGE_ID = Int64(9007199254740992)


def ages(values: List[Float64]) raises -> ColumnTable[]:
    var data = zeros[DType.float64](Shape(len(values), 1))
    for row in range(len(values)):
        data.unsafe_set(row, values[row])
    return ColumnTable(data, ["age"])


def check_storage[dtype: DType](mut table: ColumnTable[], name: String, value: Scalar[dtype]) raises:
    var values = zeros[dtype](Shape(table.rows()))
    values.unsafe_set(0, value)
    var column = categorical_column(values)
    var index = len(table.schema().names)
    table.add_categorical(name, column)
    column.set(0, None)
    var snapshot = table.copy()
    var selected = snapshot.categorical_columns[NumericCategories[dtype]]([index, index])
    selected[0].set(0, None)
    assert_equal(selected[1].get(0).or_else(Scalar[dtype](0)), value)
    assert_equal(table.categorical_columns[NumericCategories[dtype]]([index])[0].get(0).or_else(Scalar[dtype](0)), value)


def main() raises:
    var train = ages([10.0, nan[DType.float64](), 30.0, 40.0, 20.0])
    var products = zeros[DType.int64](Shape(5))
    products.unsafe_set(0, LARGE_ID + 1)
    products.unsafe_set(1, LARGE_ID)
    products.unsafe_set(2, LARGE_ID + 1)
    products.unsafe_set(3, -2)
    products.unsafe_set(4, LARGE_ID)
    train.add_categorical("product", categorical_column(products), position=0)

    var number: List[String] = ["number"]
    var category: List[String] = ["category"]
    var selector = make_column_selector(dtype_include=category^)
    assert_equal(selector(train).names, ["product"])

    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    make_column_selector(dtype_include=number^)),
        ColumnEntry("cat", ProductEncoder(sparse_output=False, handle_unknown="ignore"), selector.copy()),
    )
    var test = ages([35.0, nan[DType.float64](), 15.0])
    var new_products = zeros[DType.int64](Shape(3))
    new_products.unsafe_set(0, LARGE_ID + 2)
    new_products.unsafe_set(1, LARGE_ID)
    new_products.unsafe_set(2, LARGE_ID + 1)
    test.add_categorical("product", categorical_column(new_products))

    var copy = test.copy()
    check_storage(copy, "flag", Scalar[DType.bool](True))
    check_storage(copy, "small", Float32(1.25))
    check_storage(copy, "large", Float64(2.5))
    assert_equal(selector(copy).names, ["product", "flag", "small", "large"])
    with assert_raises(contains="different categorical domain"):
        _ = copy.categorical_columns[NumericCategories[DType.int64]]([1, 2])

    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)
    var result = ResultLine("mixed_numeric_categories")
    result.add("fit", model.fit_transform(train))
    result.add("transform", model.transform(test))
    result.add("with_extra_categories", model.transform(copy))

    var wrong = ages([35.0, 25.0, 15.0])
    var strings: List[Optional[String]] = ["a", "b", "c"]
    wrong.add_categorical("product", categorical_column(strings))
    with assert_raises(contains="different categorical domain"):
        _ = model.transform(wrong)
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
