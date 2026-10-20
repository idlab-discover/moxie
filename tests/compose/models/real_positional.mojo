"""ColumnTransformer over real Moxie transformers: positional keys and factories.

Unnamed arrays with positional keys and a passthrough remainder,
verbose_feature_names_out=False on a named table, and make_column_transformer.
Output values are compared with pinned sklearn by `real_parity.py`; feature
names are sklearn's literal results (re-derived in the parity script).
"""
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_transformer
from moxie.impute import SimpleImputer
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, zeros
from std.math import nan
from std.testing import assert_equal, assert_raises
from parity_io import ResultLine


def data[dtype: DType](rows: Int, offset: Int) raises -> NDArray[dtype]:
    """Deterministic 4-column table; NaN in columns 0 and 1 (age, fare)."""
    var X = zeros[dtype](Shape(rows, 4))
    for i in range(rows):
        var k = i + offset
        X.unsafe_set(i * 4, Float64(20 + (k * 7) % 31).cast[dtype]())
        X.unsafe_set(i * 4 + 1, (Float64((k * 13) % 17) * 3.25 + 0.5).cast[dtype]())
        X.unsafe_set(i * 4 + 2, Float64(k % 4).cast[dtype]())
        X.unsafe_set(i * 4 + 3, Float64(1 + k % 3).cast[dtype]())
        if k % 5 == 1:
            X.unsafe_set(i * 4, nan[dtype]())
        if k % 7 == 3:
            X.unsafe_set(i * 4 + 1, nan[dtype]())
    return X^


def assert_names(actual: List[String], expected: List[String]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i])


def run[dtype: DType]() raises:
    var X = data[dtype](12, 0)
    var Z = data[dtype](5, 12)
    var names: List[String] = ["age", "fare", "sibsp", "pclass"]
    var table = ColumnTable(X, names)
    var line = ResultLine(String(Int(dtype == DType.float32), "-positional"))

    # Positional selection on an unnamed array; passthrough remainder.
    var positional = ColumnTransformer(
        ColumnEntry[SimpleImputer[dtype], dtype]("num", SimpleImputer[dtype](strategy="median"), [0, 1]),
        ColumnEntry[StandardScaler[dtype], dtype]("cnt", StandardScaler[dtype](), [2]),
        remainder="passthrough",
    )
    line.add("positional", positional.fit_transform(X))
    line.add("positional_transform", positional.transform(Z))
    assert_names(positional.get_feature_names_out(),
                 ["num__x0", "num__x1", "cnt__x2", "remainder__x3"])
    var short = ColumnTransformer(
        ColumnEntry[SimpleImputer[dtype], dtype]("num", SimpleImputer[dtype](strategy="median"), [0, 1]),
        ColumnEntry[StandardScaler[dtype], dtype]("cnt", StandardScaler[dtype](), [2]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    _ = short.fit(table)
    assert_names(short.get_feature_names_out(), names.copy())

    # make_column_transformer names components after their lowercase types.
    var auto = make_column_transformer(
        ColumnEntry[SimpleImputer[dtype], dtype](SimpleImputer[dtype](strategy="median"), [0, 1]),
        ColumnEntry[StandardScaler[dtype], dtype](StandardScaler[dtype](), [2]),
    )
    line.add("auto", auto.fit_transform(X))
    assert_names(auto.get_feature_names_out(),
                 ["simpleimputer__x0", "simpleimputer__x1", "standardscaler__x2"])
    line.emit()


def main() raises:
    run[DType.float64]()
    run[DType.float32]()
