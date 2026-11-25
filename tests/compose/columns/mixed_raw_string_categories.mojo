"""Route raw strings and nullable string categories into the same encoder."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import NDArrayStrides, Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises


def ages(values: List[Float64]) raises -> ColumnTable[]:
    var data = zeros[DType.float64](Shape(len(values), 1))
    for row in range(len(values)):
        data.unsafe_set(row, values[row])
    return ColumnTable(data, ["age"])


def main() raises:
    var missing = nan[DType.float64]()
    var train = ages([10.0, missing, 30.0, 40.0, 20.0])
    var colors: List[String] = ["red", "", "bleu", "café", ""]
    train.add_strings("color", colors, position=0)
    colors[0] = "modified"
    var groups: List[Optional[String]] = ["A", None, "B", "A", None]
    train.add_categorical("group", categorical_column(groups))

    var test = ages([35.0, missing, 15.0])
    test.add_strings("group", ["B", "A", "C"], position=0)
    var test_colors: List[Optional[String]] = ["", "green", "café"]
    test.add_categorical("color", categorical_column(test_colors))
    test.add_strings("unused", ["", "", ""])

    var train_rows = zeros[DType.int64](Shape(6))
    train_rows.unsafe_set(0, 2)
    train_rows.unsafe_set(1, 4)
    train_rows.unsafe_set(2, 1)
    train_rows.unsafe_set(3, 1)
    train_rows.unsafe_set(4, 0)
    train_rows.unsafe_set(5, 3)
    train = train.take_rows(train_rows.view_with_layout(Shape(6), NDArrayStrides(-1), 5))
    var test_rows = zeros[DType.int64](Shape(4))
    test_rows.unsafe_set(0, 2)
    test_rows.unsafe_set(1, 1)
    test_rows.unsafe_set(2, 0)
    test_rows.unsafe_set(3, 2)
    test = test.take_rows(test_rows)

    var category_kinds: List[String] = ["string", "category"]
    var raw_kinds: List[String] = ["string"]
    var numeric_kinds: List[String] = ["number"]
    var selector = make_column_selector(dtype_include=category_kinds^)
    assert_equal(selector(train).names, ["color", "group"])
    assert_equal(make_column_selector(dtype_include=raw_kinds^)(train).names, ["color"])
    with assert_raises(contains="requires category storage"):
        _ = train.categorical_columns([0])
    var model = ColumnTransformer(
        ColumnEntry(
            "num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            make_column_selector(dtype_include=numeric_kinds^),
        ),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), selector^),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)

    var result = ResultLine("mixed_raw_string_categories")
    result.add("fit", model.fit_transform(train))
    result.add("transform", model.transform(test))

    var invalid = ages([35.0, missing, 15.0])
    invalid.add_categorical("group", categorical_column(zeros[DType.int64](Shape(3))))
    invalid.add_strings("color", ["", "green", "café"])
    with assert_raises(contains="different categorical domain"):
        _ = model.transform(invalid)
    with assert_raises(contains="different categorical domain"):
        _ = model.fit_transform(invalid)
    result.add("after_failed_refit", model.transform(test))

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.fit(train)
    result.add("after_fit", fresh.transform(test))

    result.emit()
    for name in model.get_feature_names_out():
        print(t"name|{name}")
    ref spans = model.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(t"span|{name}|{span.start}|{span.stop}")
