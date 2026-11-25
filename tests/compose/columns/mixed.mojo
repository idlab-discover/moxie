"""Real numeric and categorical branches of a single ColumnTransformer."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from numojo import Shape, zeros
from parity_io import ResultLine
from std.math import inf, nan
from std.testing import assert_equal, assert_false, assert_raises


def main() raises:
    var train = zeros[DType.float64](Shape(5, 1))
    var ages = List[Float64](10.0, nan[DType.float64](), 30.0, 40.0, 20.0, __list_literal__=None)
    for i in range(len(ages)):
        train.unsafe_set(i, ages[i])
    var age_name: List[String] = ["age"]
    var train_table = ColumnTable(train, age_name)
    var cells: List[Optional[String]] = ["red", "blue", "red", "amber", "blue"]
    var colors = categorical_column(cells)
    train_table.add_categorical("color", colors, position=0)
    colors.set(0, None)
    assert_equal(train_table.categorical_columns([0])[0].get(0).value(), "red")
    with assert_raises(contains="unique"):
        train_table.add_categorical("color", colors)
    with assert_raises(contains="position is out of bounds"):
        train_table.add_categorical("other", colors, position=3)
    with assert_raises(contains="cannot be borrowed"):
        _ = train_table.values()

    var include_category: List[String] = ["category"]
    var include_number: List[String] = ["number"]
    var category_selector = make_column_selector(dtype_include=include_category^)
    var number_selector = make_column_selector(dtype_include=include_number.copy())
    assert_equal(category_selector(train_table).names[0], "color")
    assert_equal(number_selector(train_table).names[0], "age")
    assert_equal(len(number_selector(train_table).names), 1)
    assert_equal(make_column_selector(dtype_exclude=include_number^)(train_table).names[0], "color")

    var numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    var model = ColumnTransformer(
        ColumnEntry("num", numeric.copy(), number_selector^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), category_selector^),
        n_jobs=2,
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(train_table)

    var test = zeros[DType.float64](Shape(3, 1))
    test.unsafe_set(0, 35.0)
    test.unsafe_set(1, nan[DType.float64]())
    test.unsafe_set(2, 15.0)
    var test_table = ColumnTable(test, age_name)
    var unseen: List[Optional[String]] = ["green", None, "red"]
    var test_colors = categorical_column(unseen)
    test_table.add_categorical("color", test_colors)
    with assert_raises(contains="same number of rows"):
        train_table.add_categorical("other", test_colors)
    assert_equal(len(train_table.schema().names), 2)

    var fresh = clone(numeric)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.with_mean = False
    _ = fresh.set_params(params)
    assert_false(fresh.estimator.get_params().with_mean)
    _ = fresh.fit(train)
    assert_equal(fresh.transform(test).shape[1], 1)

    var line = ResultLine("mixed")
    line.add("fit", model.fit_transform(train_table))
    line.add("transform", model.transform(test_table))
    with assert_raises(contains="columns are missing"):
        _ = model.transform(ColumnTable(test, age_name))
    var color_name: List[String] = ["color"]
    var wrong_kind = ColumnTable(test, color_name)
    wrong_kind.add_categorical("age", test_colors)
    test.unsafe_set(0, inf[DType.float64]())
    var invalid = ColumnTable(test, age_name)
    invalid.add_categorical("color", test_colors)
    with assert_raises():
        _ = model.fit_transform(invalid)
    with assert_raises(contains="requires numeric storage"):
        _ = model.transform(wrong_kind)
    line.add("after_failed_refit", model.transform(test_table))
    line.emit()
    for name in model.get_feature_names_out():
        print(String("name|", name))
    ref spans = model.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(String("span|", name, "|", span.start, "|", span.stop))
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    _ = cloned.fit(test_table)
    assert_equal(cloned.get_feature_names_out()[1], "cat__color_green")

    var quantile = ColumnTransformer(
        ColumnEntry("age_quantiles", QuantileTransformer(n_quantiles=7), age_name.copy()),
    )
    _ = quantile.fit_transform(train_table)
    var fitted = quantile.named_transformers_[QuantileTransformer[]]("age_quantiles")
    assert_equal(len(fitted.fit_warnings), 1)
    assert_equal(fitted.fit_warnings[0].category, "UserWarning")
