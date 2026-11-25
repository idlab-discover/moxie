"""Compare a mixed text, categorical and numeric table with pinned sklearn."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.feature_extraction.text
import sklearn.impute

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/columns"
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def reference():
    sources = {
        "compose": "sklearn/compose/_column_transformer.py",
        "pipeline": "sklearn/pipeline.py",
        "impute": "sklearn/impute/_base.py",
        "data": "sklearn/preprocessing/_data.py",
        "encoders": "sklearn/preprocessing/_encoders.py",
        "encode": "sklearn/utils/_encode.py",
        "text": "sklearn/feature_extraction/text.py",
    }
    io.verify_reference(list(sources.values()))
    io.load_pinned(sources["encode"], register=True)
    modules = {key: io.load_pinned(path, register=key == "pipeline")
               for key, path in sources.items() if key != "encode"}
    for key, cls in [
        ("compose", "ColumnTransformer"), ("pipeline", "Pipeline"),
        ("impute", "SimpleImputer"), ("data", "StandardScaler"),
        ("encoders", "OneHotEncoder"), ("text", "CountVectorizer"),
    ]:
        io.pinned_code(getattr(modules[key], cls).fit, sources[key])
    return modules


def check(modules, output, case, key, train, query):
    compose = modules["compose"]
    model = compose.ColumnTransformer([
        ("num", modules["pipeline"].make_pipeline(
            modules["impute"].SimpleImputer(strategy="median"), modules["data"].StandardScaler(),
        ), compose.make_column_selector(dtype_include="number")),
        ("cat", modules["encoders"].OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         compose.make_column_selector(dtype_include="category")),
        ("words", modules["text"].CountVectorizer(dtype=np.float64), key),
    ])
    expected_fit = model.fit_transform(train)
    expected_query = model.transform(query)
    actual = io.parse_results(output)[case]
    max_error = 0.0
    for label, expected in [
        ("fit", expected_fit), ("transform", expected_query),
        ("after_failed_refit", expected_query), ("after_fit", expected_query),
    ]:
        # Allow Float64 reduction-order rounding in the numeric block.
        error = io.close(actual[label][:, :1], expected[:, :1], 1e-12, 1e-12, label)
        max_error = max(max_error, error)
        np.testing.assert_array_equal(actual[label][:, 1:], expected[:, 1:])
    names = [line.split("|")[2] for line in output.splitlines() if line.startswith(f"name|{case}|")]
    spans = {name: slice(int(start), int(stop)) for _, _, name, start, stop in
             (line.split("|") for line in output.splitlines() if line.startswith(f"span|{case}|"))}
    assert names == model.get_feature_names_out().tolist(), names
    assert spans == model.output_indices_, spans
    print(f"{case}: count and one-hot blocks, names and spans exact, scaler max error {max_error:.3g}")


def main():
    train = pd.DataFrame({
        "color": pd.Categorical(["red", "blue", "red", "amber", "blue"]),
        "review": pd.Series(["Café tea tea", "", "green tea", "café noir", "GREEN café"], dtype="string"),
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
    })
    query = pd.DataFrame({
        "age": [35.0, np.nan, 15.0],
        "review": pd.Series(["Tea CAFÉ", "unknown", "green green"], dtype="string"),
        "color": pd.Categorical(["green", None, "red"]),
        "unused": pd.Series(["", "", ""], dtype="string"),
    })
    modules = reference()
    selector = modules["compose"].make_column_selector
    assert selector(dtype_include="string")(train) == ["review"]
    assert selector(dtype_exclude="string")(train) == ["color", "age"]
    output = io.run(ROOT, BUILD / "mixed_text", capture=True).stdout
    check(modules, output, "mixed_text_name", "review", train, query)
    check(modules, output, "mixed_text_position", 1, train, query)


if __name__ == "__main__":
    main()
