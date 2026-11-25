"""Clustering feature names in a ColumnTransformer with mixed inputs."""
from std.testing import assert_equal, assert_raises
from moxie._categorical import categorical_column
from moxie._feature_metadata import FeatureNamesOut
from moxie.cluster import Birch, BisectingKMeans, FeatureAgglomeration, MiniBatchKMeans
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.preprocessing import OneHotEncoder
from numojo import NDArray, Shape, zeros
from parity_io import ResultLine


def measurements(left: List[Float64], right: List[Float64]) raises -> NDArray[DType.float64]:
    var values = zeros[DType.float64](Shape(len(left), 2))
    for row in range(len(left)):
        values.unsafe_set(row * 2, left[row])
        values.unsafe_set(row * 2 + 1, right[row])
    return values^


def main() raises:
    comptime assert conforms_to(Birch[], FeatureNamesOut)
    comptime assert conforms_to(BisectingKMeans[], FeatureNamesOut)
    comptime assert conforms_to(MiniBatchKMeans[], FeatureNamesOut)
    var train_names: List[String] = ["left", "right"]
    var train = ColumnTable(
        measurements([2.0, 4.0, 8.0, 16.0], [4.0, 6.0, 10.0, 18.0]), train_names,
    )
    var colors: List[Optional[String]] = ["blue", "red", "blue", "red"]
    train.add_categorical("color", categorical_column(colors), position=1)
    var test_names: List[String] = ["right", "left"]
    var test = ColumnTable(measurements([8.0, 12.0], [4.0, 8.0]), test_names)
    var test_colors: List[Optional[String]] = ["red", "green"]
    test.add_categorical("color", categorical_column(test_colors), position=0)

    var numeric_dtypes: List[String] = ["number"]
    var categorical_names: List[String] = ["color"]
    var numeric_columns = make_column_selector(dtype_include=numeric_dtypes^)
    var model = ColumnTransformer(
        ColumnEntry("pool", FeatureAgglomeration(n_clusters=1), numeric_columns^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_names^),
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    var line = ResultLine("mixed_cluster_names")
    line.add("fit", model.fit_transform(train))
    line.add("transform", model.transform(test))
    var missing_color = ColumnTable(measurements([2.0], [4.0]), train_names)
    with assert_raises(contains="not a column"):
        _ = model.fit_transform(missing_color)
    line.add("after_failed_refit", model.transform(test))
    var names = model.get_feature_names_out(train.schema().names)
    assert_equal(len(names), 3)
    assert_equal(names[0], "pool__featureagglomeration0")
    line.emit()
    for name in names:
        print(String("name|", name))
    ref spans = model.output_indices_.value()
    for name in ["pool", "cat", "remainder"]:
        ref span = spans[name]
        print(String("span|", name, "|", span.start, "|", span.stop))
