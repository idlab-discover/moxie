"""Compare mixed tables with the pinned sklearn compositions.

One-hot values, names and spans must match exactly. The scaler uses Float64
accumulation and can round differently from NumPy's reductions, so its output
allows 1e-12 absolute and relative error.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import sklearn.compose  # Load wheel support modules before replacing pinned modules.
import sklearn.impute

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/columns"
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

COMPOSE = "sklearn/compose/_column_transformer.py"
PIPELINE = "sklearn/pipeline.py"
IMPUTE = "sklearn/impute/_base.py"
DATA = "sklearn/preprocessing/_data.py"
ENCODERS = "sklearn/preprocessing/_encoders.py"
ENCODE = "sklearn/utils/_encode.py"
hashes = io.verify_reference([COMPOSE, PIPELINE, IMPUTE, DATA, ENCODERS, ENCODE])
encoding = io.load_pinned(ENCODE, register=True)
pipeline = io.load_pinned(PIPELINE, register=True)
compose = io.load_pinned(COMPOSE)
impute = io.load_pinned(IMPUTE)
data = io.load_pinned(DATA)
encoders = io.load_pinned(ENCODERS)
for method, source in [
    (encoding._encode, ENCODE),
    (compose.ColumnTransformer.fit_transform, COMPOSE),
    (pipeline.Pipeline.fit_transform, PIPELINE),
    (impute.SimpleImputer.fit, IMPUTE),
    (data.StandardScaler.fit, DATA),
    (encoders.OneHotEncoder.fit, ENCODERS),
]:
    io.pinned_code(method, source)

def check_mixed(case, column, train_categories, test_categories, *, extra_outputs=()):
    train = pd.DataFrame({
        column: pd.Categorical(train_categories),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
    })
    test = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        column: pd.Categorical(test_categories),
    })
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         compose.make_column_selector(dtype_include="number")),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         compose.make_column_selector(dtype_include="category")),
    ])
    check_composition(case, model, train, test, extra_outputs)


def check_composition(case, model, train, test, extra_outputs=(), *, numeric_width=1):
    expected_fit = model.fit_transform(train)
    expected_transform = model.transform(test)
    output = io.run(ROOT, BUILD / case, capture=True).stdout
    actual = io.parse_results(output)[case]
    errors = {}
    for label, expected in [
        ("fit", expected_fit),
        ("transform", expected_transform),
        ("after_failed_refit", expected_transform),
        *((label, expected_transform) for label in extra_outputs),
    ]:
        errors[label] = io.close(
            actual[label][:, :numeric_width], expected[:, :numeric_width], 1e-12, 1e-12, label,
        )
        np.testing.assert_array_equal(actual[label][:, numeric_width:], expected[:, numeric_width:])
    names = [line.split("|")[1] for line in output.splitlines() if line.startswith("name|")]
    spans = {name: slice(int(start), int(stop)) for _, name, start, stop in
             (line.split("|") for line in output.splitlines() if line.startswith("span|"))}
    assert names == model.get_feature_names_out().tolist(), names
    assert spans == model.output_indices_, spans

    report = BUILD / "parity"
    report.mkdir(parents=True, exist_ok=True)
    (report / f"{case}-report.json").write_text(json.dumps({
        "revision": io.REVISION, "sources": hashes, "scaler_max_errors": errors,
        "names": names, "spans": {name: [span.start, span.stop] for name, span in spans.items()},
    }, indent=2) + "\n")
    print(f"{case}: one-hot values, names and spans exact, scaler within 1e-12")


def check_categorical_imputation(case, *, numeric=False):
    categories = {
        "amber": 1.5, "red": 2.5, "circle": 3.5,
        "square": 4.5, "triangle": 5.5, "green": 6.5,
        "ignored": 1.5, "value": 2.5,
    }

    def categorical(values):
        if numeric:
            values = [categories[value] if value is not None else np.nan for value in values]
        return pd.Categorical(values)

    train = pd.DataFrame({
        "color": categorical(["red", None, "red", "amber", None]),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
        "empty": categorical([None] * 5),
        "shape": categorical(["square", "circle", None, "square", "circle"]),
    })
    test = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        "shape": categorical([None, "triangle", "square"]),
        "color": categorical(["green", None, "amber"]),
        "empty": categorical(["ignored", None, "value"]),
    })
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         compose.make_column_selector(dtype_include="number")),
        ("cat", pipeline.make_pipeline(
            impute.SimpleImputer(strategy="most_frequent"),
            encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        ), compose.make_column_selector(dtype_include="category")),
    ])
    check_composition(case, model, train, test, ("after_fit",))


def check_mixed_float_columns():
    train = pd.DataFrame({
        "rating": np.array([1.25, 2.5, np.nan, 1.25, 5.0], dtype=np.float32),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
        "color": pd.Categorical(["red", "blue", "red", "amber", "blue"]),
    })
    test = pd.DataFrame({
        "color": pd.Categorical(["green", None, "red"]),
        "age": [35.0, np.nan, 15.0],
        "rating": np.array([1.25, np.nan, 2.5], dtype=np.float32),
    })
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         compose.make_column_selector(dtype_include="number")),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         compose.make_column_selector(dtype_include="category")),
    ])
    check_composition(
        "mixed_float_columns", model, train, test, ("after_fit", "after_dtype_change"),
        numeric_width=2,
    )


def check_mixed_integer_columns():
    train = pd.DataFrame({
        "count": np.array([-4, 0, 7, 0, 13], dtype=np.int64),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
        "color": pd.Categorical(["red", "blue", "red", "amber", "blue"]),
    })
    test = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        "color": pd.Categorical(["green", None, "red"]),
        "count": np.array([0, -4, 2**53 + 1], dtype=np.int64),
    })
    for kind in ["int", "int64", "integer", "signedinteger"]:
        assert compose.make_column_selector(dtype_include=kind)(train) == ["count"]
    assert compose.make_column_selector(dtype_include="floating")(train) == ["age"]
    assert compose.make_column_selector(dtype_include="unsignedinteger")(train) == []
    assert compose.make_column_selector(dtype_include="number", dtype_exclude="integer")(train) == ["age"]
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         compose.make_column_selector(dtype_include="number")),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         compose.make_column_selector(dtype_include="category")),
    ])
    check_composition(
        "mixed_integer_columns", model, train, test, ("after_fit", "after_dtype_change"),
        numeric_width=2,
    )


def check_mixed_boolean_columns():
    train = pd.DataFrame({
        "active": np.array([True, False, True, False, True], dtype=bool),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
        "color": pd.Categorical(["red", "blue", "red", "amber", "blue"]),
    })
    test = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        "color": pd.Categorical(["green", None, "red"]),
        "active": np.array([False, True, False], dtype=bool),
    })
    selector = compose.make_column_selector
    dense_kinds = ["number", "bool"]
    assert selector(dtype_include="bool")(train) == ["active"]
    assert selector(dtype_include="number")(train) == ["age"]
    assert selector(dtype_include=dense_kinds)(train) == ["active", "age"]
    assert selector(dtype_exclude="bool")(train) == ["age", "color"]
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         selector(dtype_include=dense_kinds)),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         selector(dtype_include="category")),
    ])
    check_composition(
        "mixed_boolean_columns", model, train, test, ("after_fit",), numeric_width=2,
    )


def check_mixed_numeric_categories():
    large = 2**53
    train = pd.DataFrame({
        "product": np.array([large + 1, large, large + 1, -2, large], dtype=np.int64),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
    })
    test = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        "product": np.array([large + 2, large, large + 1], dtype=np.int64),
    })
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
         compose.make_column_selector(dtype_include="floating")),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         compose.make_column_selector(dtype_include=["int64", "category"])),
    ])
    check_composition(
        "mixed_numeric_categories", model, train, test, ("with_extra_categories", "after_fit"),
    )


def check_mixed_raw_float_categories():
    train = pd.DataFrame({
        "rating": [1.25, np.nan, 2.5, 1.25, 2.5],
        "band": [3.5, 4.5, 3.5, 4.5, 3.5],
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
    })
    test = pd.DataFrame({
        "band": [4.5, 3.5, 9.5],
        "age": [35.0, np.nan, 15.0],
        "rating": [7.5, np.nan, 2.5],
    })
    model = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()), ["age"]),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["rating", "band"]),
    ])
    check_composition("mixed_raw_float_categories", model, train, test)


if __name__ == "__main__":
    check_mixed("mixed", "color", ["red", "blue", "red", "amber", "blue"], ["green", None, "red"])
    check_mixed_numeric_categories()
    check_mixed_raw_float_categories()
    check_categorical_imputation("mixed_categorical_impute")
    check_categorical_imputation("mixed_numeric_impute", numeric=True)
    check_mixed_float_columns()
    check_mixed_integer_columns()
    check_mixed_boolean_columns()
