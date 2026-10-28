"""ColumnTransformer over real Moxie impute/preprocessing transformers.

Mirrors the numeric branch of sklearn's mixed-types example: SimpleImputer
(median) and StandardScaler on named columns of data with NaN, plus an
estimator remainder. `real_positional.mojo` covers positional keys and
make_column_transformer (split to stay under the compiler memory cap).
Output values are compared with pinned sklearn by `real_parity.py`; feature
names and spans are sklearn's literal results (re-derived there).
"""
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.impute import SimpleImputer
from moxie.preprocessing import MinMaxScaler, StandardScaler
from numojo import NDArray, Shape, zeros
from std.math import nan
from std.testing import assert_equal, assert_false, assert_true, assert_raises
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
    var numeric: List[String] = ["age", "fare"]
    var counts: List[String] = ["sibsp"]
    var named = ColumnTransformer(
        ColumnEntry[SimpleImputer[dtype], dtype]("num", SimpleImputer[dtype](strategy="median"), numeric.copy()),
        ColumnEntry[StandardScaler[dtype], dtype]("cnt", StandardScaler[dtype](), counts.copy()),
        remainder=MinMaxScaler[dtype](),
    )
    with assert_raises():
        _ = named.get_feature_names_out()
    var table = ColumnTable(X, names)
    var fitted = named.fit_transform(table)
    assert_names(named.get_feature_names_out(),
                 ["num__age", "num__fare", "cnt__sibsp", "remainder__pclass"])
    assert_names(named.feature_names_in_.value(), names.copy())
    assert_equal(named.output_indices_.value()["num"].stop, 2)
    assert_equal(named.output_indices_.value()["cnt"].start, 2)
    assert_equal(named.output_indices_.value()["remainder"].start, 3)
    var imputer = named.named_transformers_[SimpleImputer[dtype]]("num")
    var line = ResultLine(String(Int(dtype == DType.float32), "-named"))
    line.add("fit_transform", fitted)
    line.add("transform", named.transform(ColumnTable(Z, names)))
    line.add("statistics", imputer.statistics_)

    # Clone drops learned state; a refit on new data replaces it.
    var fresh = clone(named)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_true(named.__sklearn_is_fitted__())
    _ = fresh.fit(ColumnTable(Z, names))
    line.add("refit_transform", fresh.transform(table))
    line.emit()
