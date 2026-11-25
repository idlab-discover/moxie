"""Ordinal and cross-fitted target encoding on a mixed ColumnTable."""
from moxie._categorical import categorical_column
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.model_selection import StratifiedKFold
from moxie.preprocessing import OrdinalEncoder, TargetEncoder, TargetEncoderSplits
from numojo import Shape, zeros
from parity_io import ResultLine
from std.testing import assert_equal, assert_false, assert_raises


def table(ages: List[Float64], colors: List[Optional[String]], color_first: Bool) raises -> ColumnTable[]:
    var data = zeros[DType.float64](Shape(len(ages), 1))
    for row in range(len(ages)):
        data.unsafe_set(row, ages[row])
    var result = ColumnTable(data, ["age"])
    result.add_categorical("color", categorical_column(colors), position=0 if color_first else 1)
    return result^


def main() raises:
    var train = table([10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                      ["red", "blue", "amber", "blue", "red", "amber"], True)
    var test = table([15.0, 35.0, 55.0], ["green", None, "red"], False)
    var y = zeros[DType.float64](Shape(6))
    var classes = zeros[DType.int64](Shape(6))
    for row in range(6):
        y.unsafe_set(row, Float64(row % 3))
        classes.unsafe_set(row, Int64(row % 3))
    var cv = TargetEncoderSplits(StratifiedKFold(2).split(6, classes^))
    var color: List[String] = ["color"]
    var age: List[String] = ["age"]
    var model = ColumnTransformer(
        ColumnEntry("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1.0), color.copy()),
        ColumnEntry("target", TargetEncoder(smooth=0.0, cv=cv^), color^),
        ColumnEntry("numeric", "passthrough", age^),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(test)
    with assert_raises(contains="TargetEncoder requires y"):
        _ = model.fit_transform(train)
    assert_false(model.__sklearn_is_fitted__())

    var training = model.fit_transform(train, y.copy())
    var fitted_training = model.transform(train)
    # The first red row belongs to class 0. Its out-of-fold encoding excludes itself.
    assert_equal(training.unsafe_get(1), 0.0)
    assert_equal(fitted_training.unsafe_get(1), 0.5)
    var line = ResultLine("mixed_encoders")
    line.add("fit", training)
    line.add("transform", model.transform(test))
    with assert_raises(contains="TargetEncoder requires y"):
        _ = model.fit(train)
    line.add("after_failed_refit", model.transform(test))

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.fit_transform(train, y.copy()).unsafe_get(1), 0.0)
    _ = fresh.fit(train, y^)
    line.add("after_fit", fresh.transform(test))
    line.emit()
    for name in model.get_feature_names_out():
        print(String("name|", name))
    ref spans = model.output_indices_.value()
    for name in ["ordinal", "target", "numeric", "remainder"]:
        ref span = spans[name]
        print(String("span|", name, "|", span.start, "|", span.stop))
