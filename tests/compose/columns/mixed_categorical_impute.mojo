"""Impute nullable strings before one-hot encoding alongside numeric preprocessing."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import CategoricalStringImputer, SimpleImputer, StringMatrix, StringSimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises


def ages(values: List[Float64]) raises -> ColumnTable[]:
    var data = zeros[DType.float64](Shape(len(values), 1))
    for row in range(len(values)):
        data.unsafe_set(row, values[row])
    return ColumnTable(data, ["age"])


def check_adapter() raises:
    var cells: List[Optional[String]] = ["blue", None, "red"]
    var matrix = StringMatrix(cells, 3, 1)
    var source = StringSimpleImputer(strategy="most_frequent")
    _ = source.fit(matrix)
    var stage = source.as_categorical()
    assert_false(stage.__sklearn_is_fitted__())
    var column = categorical_column(cells)
    var X: CategoricalStringImputer.Input = [column.copy()]
    with assert_raises(contains="NotFittedError"):
        _ = stage.transform(X)
    assert_equal(stage.fit_transform(X)[0].get(1).or_else(""), "blue")
    assert_false(Bool(column.get(1)))

    var short_cells: List[Optional[String]] = ["green"]
    var short = categorical_column(short_cells)
    with assert_raises(contains="matching row counts"):
        _ = stage.fit_transform([column.copy(), short.copy()])
    assert_equal(stage.transform(X)[0].get(1).or_else(""), "blue")
    with assert_raises(contains="feature count differs"):
        _ = stage.transform([column.copy(), column.copy()])
    assert_false(clone(stage).__sklearn_is_fitted__())

    _ = stage.fit([short.copy()])
    assert_equal(stage.transform(X)[0].get(1).or_else(""), "green")
    var config = stage.get_params()
    config.strategy = "constant"
    config.fill_value = "amber"
    _ = stage.set_params(config)
    assert_false(stage.__sklearn_is_fitted__())
    assert_equal(stage.fit_transform(X)[0].get(1).or_else(""), "amber")
    config.add_indicator = True
    _ = stage.set_params(config)
    with assert_raises(contains="cannot include Boolean indicators"):
        _ = stage.fit(X)
    assert_false(stage.__sklearn_is_fitted__())


def main() raises:
    check_adapter()
    var train = ages([10.0, nan[DType.float64](), 30.0, 40.0, 20.0])
    var colors: List[Optional[String]] = ["red", None, "red", "amber", None]
    var empty: List[Optional[String]] = [None, None, None, None, None]
    var shapes: List[Optional[String]] = ["square", "circle", None, "square", "circle"]
    train.add_categorical("color", categorical_column(colors), position=0)
    train.add_categorical("empty", categorical_column(empty))
    train.add_categorical("shape", categorical_column(shapes))

    var query = ages([35.0, nan[DType.float64](), 15.0])
    var query_shapes: List[Optional[String]] = [None, "triangle", "square"]
    var query_colors: List[Optional[String]] = ["green", None, "amber"]
    var query_empty: List[Optional[String]] = ["ignored", None, "value"]
    query.add_categorical("shape", categorical_column(query_shapes))
    query.add_categorical("color", categorical_column(query_colors))
    query.add_categorical("empty", categorical_column(query_empty))

    var number: List[String] = ["number"]
    var category: List[String] = ["category"]
    var categorical = make_pipeline(
        StringSimpleImputer(strategy="most_frequent").as_categorical(),
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )
    assert_equal(categorical.transformer_name, "simpleimputer")
    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    make_column_selector(dtype_include=number^)),
        ColumnEntry("cat", categorical^, make_column_selector(dtype_include=category^)),
    )
    var result = ResultLine("mixed_categorical_impute")
    result.add("fit", model.fit_transform(train))
    result.add("transform", model.transform(query))

    var wrong = ages([35.0, 25.0, 15.0])
    wrong.add_strings("color", ["red", "red", "amber"])
    wrong.add_categorical("shape", categorical_column(query_shapes))
    wrong.add_categorical("empty", categorical_column(query_empty))
    with assert_raises(contains="requires category storage"):
        _ = model.transform(wrong)
    # Fit selectors would omit the string column. In this refit the remaining
    # categorical features are all empty, which leaves no features to encode.
    var invalid = ages([35.0, 25.0, 15.0])
    var absent: List[Optional[String]] = [None, None, None]
    invalid.add_categorical("color", categorical_column(absent))
    with assert_raises(contains="0 feature(s)"):
        _ = model.fit_transform(invalid)
    result.add("after_failed_refit", model.transform(query))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.fit(train)
    result.add("after_fit", fresh.transform(query))
    result.emit()
    for name in model.get_feature_names_out():
        print(String("name|", name))
    ref spans = model.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(String("span|", name, "|", span.start, "|", span.stop))
