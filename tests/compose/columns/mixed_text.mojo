"""Raw text, categorical and numeric branches in one native ColumnTransformer."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnKey, ColumnTable, ColumnTransformer, Columns, make_column_selector
from moxie.feature_extraction.text import CountVectorizer
from moxie.impute import SimpleImputer
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


def main() raises:
    var train = ages([10.0, nan[DType.float64](), 30.0, 40.0, 20.0])
    var train_colors: List[Optional[String]] = ["red", "blue", "red", "amber", "blue"]
    train.add_categorical("color", categorical_column(train_colors), position=0)
    var docs: List[String] = ["Café tea tea", "", "green tea", "café noir", "GREEN café"]
    train.add_strings("review", docs, position=1)
    docs[0] = "modified"
    var snapshot = train.copy()
    var selected = snapshot.string_column(1)
    assert_equal(selected[0], "Café tea tea")
    assert_equal(selected[1], "")
    selected[0] = "modified again"
    assert_equal(snapshot.string_column(1)[0], "Café tea tea")
    assert_equal(train.schema().dtypes, ["category", "string", "float64"])

    var number: List[String] = ["number"]
    var category: List[String] = ["category"]
    var strings: List[String] = ["string"]
    var selector = make_column_selector(dtype_include=strings.copy())
    assert_equal(selector(train).names, ["review"])
    assert_equal(make_column_selector(dtype_exclude=strings^)(train).names, ["color", "age"])
    with assert_raises(contains="same number of rows"):
        train.add_strings("short", ["one"])
    with assert_raises(contains="unique"):
        train.add_strings("review", docs)
    with assert_raises(contains="out of bounds"):
        train.add_strings("invalid", docs, position=4)
    with assert_raises(contains="out of bounds"):
        _ = train.string_column(-1)
    with assert_raises(contains="requires string storage"):
        _ = train.string_column(0)
    with assert_raises(contains="mixed ColumnTable"):
        _ = train.values()

    # List and dtype selectors retain rank two, including one-element results.
    var keys: List[ColumnKey] = [ColumnKey(selector.copy()), ColumnKey(String("review")), ColumnKey(1)]
    for index in range(len(keys)):
        ref key = keys[index]
        var model = ColumnTransformer(
            ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                        make_column_selector(dtype_include=number.copy())),
            ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        make_column_selector(dtype_include=category.copy())),
            ColumnEntry("words", CountVectorizer[DType.float64](), Columns[DType.float64](key.copy())),
        )
        if not key.is_scalar():
            with assert_raises(contains="require a scalar column key"):
                _ = model.fit_transform(train)
            continue

        var query = ages([35.0, nan[DType.float64](), 15.0])
        query.add_strings("review", ["Tea CAFÉ", "unknown", "green green"])
        var query_colors: List[Optional[String]] = ["green", None, "red"]
        query.add_categorical("color", categorical_column(query_colors))
        query.add_strings("unused", ["", "", ""])

        with assert_raises(contains="NotFittedError"):
            _ = model.transform(query)
        var fitted = model.fit_transform(train)
        var transformed = model.transform(query)
        var wrong = ages([35.0, 25.0, 15.0])
        wrong.add_categorical("color", categorical_column(query_colors), position=0)
        wrong.add_categorical("review", categorical_column(query_colors), position=1)
        with assert_raises(contains="requires string storage"):
            _ = model.transform(wrong)
        with assert_raises(contains="requires string storage"):
            _ = model.fit_transform(wrong)

        var case_name = String("mixed_text_name" if index == 1 else "mixed_text_position")
        var result = ResultLine(case_name)
        result.add("fit", fitted)
        result.add("transform", transformed)
        result.add("after_failed_refit", model.transform(query))
        var fresh = clone(model)
        assert_false(fresh.__sklearn_is_fitted__())
        _ = fresh.fit(train)
        result.add("after_fit", fresh.transform(query))
        result.emit()
        for name in model.get_feature_names_out():
            print(String("name|", case_name, "|", name))
        ref spans = model.output_indices_.value()
        for name in ["num", "cat", "words", "remainder"]:
            ref span = spans[name]
            print(String("span|", case_name, "|", name, "|", span.start, "|", span.stop))
