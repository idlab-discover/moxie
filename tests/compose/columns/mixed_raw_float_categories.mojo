"""Encode same-dtype raw values from the matrix and added columns."""
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from moxie.preprocessing._vocabulary import NumericKind
from numojo import NDArrayStrides, Shape, zeros
from parity_io import ResultLine
from std.math import inf, nan
from std.testing import assert_raises


def table(
    ages: List[Float64], ratings: List[Float64], bands: List[Float64], rating_first: Bool,
) raises -> ColumnTable[]:
    var rows = len(ages)
    var backing = zeros[DType.float64](Shape(rows * 3))
    var band = zeros[DType.float64](Shape(rows))
    for row in range(rows):
        var offset = (rows - 1 - row) * 3
        backing.unsafe_set(offset + 1, ratings[row] if rating_first else ages[row])
        backing.unsafe_set(offset + 2, ages[row] if rating_first else ratings[row])
        band.unsafe_set(row, bands[row])
    var data = backing.view_with_layout(Shape(rows, 2), NDArrayStrides(-3, 1), (rows - 1) * 3 + 1)
    var names: List[String] = ["rating", "age"] if rating_first else ["age", "rating"]
    var result = ColumnTable(data, names)
    result.add_numeric("band", band, position=1 if rating_first else 0)
    return result^


def main() raises:
    var missing = nan[DType.float64]()
    var train = table(
        [10.0, missing, 30.0, 40.0, 20.0],
        [1.25, missing, 2.5, 1.25, 2.5],
        [3.5, 4.5, 3.5, 4.5, 3.5], True,
    )
    var test = table([35.0, missing, 15.0], [7.5, missing, 2.5], [4.5, 3.5, 9.5], False)
    var numeric: List[String] = ["age"]
    var categorical: List[String] = ["rating", "band"]
    var model = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric^),
        ColumnEntry(
            "cat", OneHotEncoder[NumericKind[DType.float64]](sparse_output=False, handle_unknown="ignore"),
            categorical^,
        ),
    )
    var result = ResultLine("mixed_raw_float_categories")
    result.add("fit", model.fit_transform(train))
    result.add("transform", model.transform(test))
    var invalid = table([35.0], [inf[DType.float64]()], [4.5], False)
    with assert_raises(contains="categorical values must be finite or missing"):
        _ = model.fit_transform(invalid)
    result.add("after_failed_refit", model.transform(test))
    result.emit()
    for name in model.get_feature_names_out():
        print(t"name|{name}")
    ref spans = model.output_indices_.value()
    for name in ["num", "cat", "remainder"]:
        ref span = spans[name]
        print(t"span|{name}|{span.start}|{span.stop}")
