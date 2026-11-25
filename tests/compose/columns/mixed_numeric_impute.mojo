"""Impute nullable numeric categories before one-hot encoding alongside numeric preprocessing."""
from moxie._categorical import CategoricalColumn, NumericCategories, categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import CategoricalNumericImputer, SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from moxie.preprocessing._vocabulary import NumericKind
from numojo import Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises


def ages(values: List[Float64]) raises -> ColumnTable[]:
    var data = zeros[DType.float64](Shape(len(values), 1))
    for row in range(len(values)):
        data.unsafe_set(row, values[row])
    return ColumnTable(data, ["age"])


def categories(values: List[Float64]) raises -> CategoricalColumn[NumericCategories[DType.float64]]:
    var data = zeros[DType.float64](Shape(len(values)))
    for row in range(len(values)):
        data.unsafe_set(row, values[row])
    return categorical_column(data)


def main() raises:
    var missing = nan[DType.float64]()
    var train = ages([10.0, nan[DType.float64](), 30.0, 40.0, 20.0])
    var colors: List[Float64] = [2.5, missing, 2.5, 1.5, missing]
    var empty: List[Float64] = [missing, missing, missing, missing, missing]
    var shapes: List[Float64] = [4.5, 3.5, missing, 4.5, 3.5]
    train.add_categorical("color", categories(colors), position=0)
    train.add_categorical("empty", categories(empty))
    train.add_categorical("shape", categories(shapes))

    var query = ages([35.0, nan[DType.float64](), 15.0])
    var query_shapes: List[Float64] = [missing, 5.5, 4.5]
    var query_colors: List[Float64] = [6.5, missing, 1.5]
    var query_empty: List[Float64] = [1.5, missing, 2.5]
    query.add_categorical("shape", categories(query_shapes))
    query.add_categorical("color", categories(query_colors))
    query.add_categorical("empty", categories(query_empty))

    var number: List[String] = ["number"]
    var category: List[String] = ["category"]
    var categorical = make_pipeline(
        CategoricalNumericImputer[DType.float64](),
        OneHotEncoder[NumericKind[DType.float64]](sparse_output=False, handle_unknown="ignore"),
    )
    assert_equal(categorical.transformer_name, "simpleimputer")
    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    make_column_selector(dtype_include=number^)),
        ColumnEntry("cat", categorical^, make_column_selector(dtype_include=category^)),
    )
    var result = ResultLine("mixed_numeric_impute")
    result.add("fit", model.fit_transform(train))
    result.add("transform", model.transform(query))

    var wrong = ages([35.0, 25.0, 15.0])
    var strings: List[Optional[String]] = ["a", "a", "b"]
    wrong.add_categorical("color", categorical_column(strings))
    wrong.add_categorical("shape", categories(query_shapes))
    wrong.add_categorical("empty", categories(query_empty))
    with assert_raises(contains="different categorical domain"):
        _ = model.transform(wrong)
    # An all-missing categorical branch leaves no features to encode.
    var invalid = ages([35.0, 25.0, 15.0])
    var absent: List[Float64] = [missing, missing, missing]
    invalid.add_categorical("color", categories(absent))
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
