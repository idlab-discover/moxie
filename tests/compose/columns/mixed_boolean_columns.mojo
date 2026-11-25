"""Raw Boolean features alongside numeric and categorical preprocessing."""
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


def table(
    flags: List[Bool], ages: List[Float64], colors: List[Optional[String]],
    flag_first: Bool,
) raises -> ColumnTable[]:
    var age = zeros[DType.float64](Shape(len(ages), 1))
    var backing = zeros[DType.bool](Shape(2 * len(flags)))
    for row in range(len(ages)):
        age.unsafe_set(row, ages[row])
        backing.unsafe_set(backing.size - 1 - 2 * row, flags[row])
    var flags_view = backing.view_with_layout(
        Shape(len(flags)), NDArrayStrides(-2), backing.size - 1,
    )
    var result = ColumnTable(age, ["age"])
    result.add_categorical("color", categorical_column(colors))
    result.add_numeric("active", flags_view, position=0 if flag_first else 2)
    backing.unsafe_set(backing.size - 1, not flags[0])
    return result^


def main() raises:
    var train = table(
        [True, False, True, False, True],
        [10.0, nan[DType.float64](), 30.0, 40.0, 20.0],
        ["red", "blue", "red", "amber", "blue"], True,
    )
    var test = table(
        [False, True, False], [35.0, nan[DType.float64](), 15.0],
        ["green", None, "red"], False,
    )
    assert_equal(train.schema().dtypes, ["bool", "float64", "category"])
    var bool_kinds: List[String] = ["bool"]
    var number_kinds: List[String] = ["number"]
    var dense_kinds: List[String] = ["number", "bool"]
    var category_kinds: List[String] = ["category"]
    assert_equal(make_column_selector(dtype_include=bool_kinds.copy())(train).names, ["active"])
    assert_equal(make_column_selector(dtype_include=number_kinds^)(train).names, ["age"])
    var dense = make_column_selector(dtype_include=dense_kinds^)
    assert_equal(dense(train).names, ["active", "age"])
    assert_equal(
        make_column_selector(dtype_exclude=bool_kinds^)(train).names,
        ["age", "color"],
    )
    with assert_raises(contains="rank-1"):
        train.add_numeric("invalid", zeros[DType.bool](Shape(5, 1)))
    with assert_raises(contains="same number of rows"):
        train.add_numeric("invalid", zeros[DType.bool](Shape(3)))
    with assert_raises(contains="requires category storage"):
        _ = train.categorical_columns([0])
    assert_equal(train.schema().names, ["active", "age", "color"])

    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), dense^),
        ColumnEntry(
            "cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            make_column_selector(dtype_include=category_kinds^),
        ),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)
    var result = ResultLine("mixed_boolean_columns")
    result.add("fit", model.fit_transform(train.copy()))
    result.add("transform", model.transform(test))

    var wrong = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    var flags = zeros[DType.bool](Shape(3))
    wrong.add_categorical("active", categorical_column(flags))
    wrong.add_categorical("color", categorical_column(flags))
    with assert_raises(contains="requires numeric storage"):
        _ = model.transform(wrong)
    wrong.add_numeric("flag", flags)
    with assert_raises(contains="different categorical domain"):
        _ = model.fit_transform(wrong)
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
