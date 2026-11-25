"""Use a mixed ColumnTable as the input of a native predictive pipeline."""
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.impute import SimpleImputer
from moxie.linear_model import Ridge
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import NDArray, Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def vector(values: List[Float64]) raises -> NDArray[DType.float64]:
    var result = zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def table(ages: List[Float64], colors: List[String], color_first: Bool = False) raises -> ColumnTable[]:
    var result = ColumnTable(vector(ages).reshape(Shape(len(ages), 1)), ["age"])
    result.add_strings("color", colors, position=0 if color_first else 1)
    return result^


def main() raises:
    var missing = nan[DType.float64]()
    var train = table([10, missing, 30, 40, 20], ["red", "blue", "red", "amber", "blue"])
    var test = table([35, missing, 15], ["green", "blue", "red"], color_first=True)
    var y = vector([2, -1, 5, 7, 0])
    var numeric: List[String] = ["age"]
    var categorical: List[String] = ["color"]
    var columns = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical^),
    )
    var pipe = make_pipeline(columns^.as_table_transformer(), Ridge())
    assert_equal(pipe.transformer_name, "columntransformer")
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict(test)

    _ = pipe.fit(train, y)
    assert_equal(pipe.n_features_in_(), 2)
    var result = ResultLine("mixed_ridge")
    result.add("transform", pipe.transformer.transform(test))
    result.add("predict", pipe.predict(test))
    result.scalar("score", pipe.score(test, vector([5, 0, 2])))
    for name in pipe.transformer.get_feature_names_out():
        print(t"name|{name}")
    ref spans = pipe.transformer.columns.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(t"span|{name}|{span.start}|{span.stop}")

    var invalid = ColumnTable(zeros[DType.float64](Shape(3, 1)), ["age"])
    with assert_raises(contains="columns are missing from the named input"):
        _ = pipe.predict(invalid)
    with assert_raises(contains="is not a column of the dataframe"):
        _ = pipe.fit(invalid, vector([0, 0, 0]))
    result.add("after_failed_refit", pipe.predict(test))

    var fresh = clone(pipe)
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.alpha = Float64(3)
    _ = fresh.set_params(params)
    _ = fresh.fit(train, y)
    result.add("changed_alpha", fresh.predict(test))
    result.add("original", pipe.predict(test))
    _ = pipe.fit(train, vector([4, -2, 10, 14, 0]))
    result.add("refit", pipe.predict(test))

    var fitted_columns = pipe.transformer.columns.copy()
    var adapted = fitted_columns^.as_table_transformer()
    assert_true(adapted.__sklearn_is_fitted__())
    result.add("adapted_transform", adapted.transform(test))
    var standalone = clone(adapted)
    _ = standalone.fit(train)
    result.add("adapted_fit", standalone.transform(test))
    result.emit()
