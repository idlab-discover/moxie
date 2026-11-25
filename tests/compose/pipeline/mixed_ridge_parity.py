"""Compare a predictive pipeline on mixed tables against pinned scikit-learn."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "columns"))
from mixed_parity import ROOT, compose, data, encoders, impute, io, pipeline


def main():
    ridge_base = "sklearn/linear_model/_base.py"
    ridge_source = "sklearn/linear_model/_ridge.py"
    io.verify_reference([ridge_base, ridge_source])
    io.load_pinned(ridge_base, register=True)
    ridge = io.load_pinned(ridge_source)
    io.pinned_code(ridge.Ridge.fit, ridge_source)

    train = pd.DataFrame({
        "age": [10.0, np.nan, 30.0, 40.0, 20.0],
        "color": ["red", "blue", "red", "amber", "blue"],
    })
    test = pd.DataFrame({
        "color": ["green", "blue", "red"],
        "age": [35.0, np.nan, 15.0],
    })
    y = np.array([2, -1, 5, 7, 0], dtype=np.float64)
    columns = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()), ["age"]),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["color"]),
    ])
    model = pipeline.make_pipeline(columns, ridge.Ridge()).fit(train, y)
    assert model.n_features_in_ == 2
    prediction = model.predict(test)
    transformed = columns.transform(test)
    expected = {
        "transform": transformed,
        "predict": prediction,
        "score": model.score(test, np.array([5, 0, 2])),
        # Native ColumnTransformer preserves its fitted state on a schema error.
        "after_failed_refit": prediction,
        "changed_alpha": clone(model).set_params(
            ridge__alpha=3.0,
        ).fit(train, y).predict(test),
        "original": prediction,
        "refit": model.fit(train, 2 * y).predict(test),
        "adapted_transform": transformed,
        "adapted_fit": transformed,
    }
    binary = ROOT / "builds/tests/compose/pipeline/mixed_ridge"
    output = io.run(ROOT, binary, capture=True).stdout
    actual = io.parse_results(output)["mixed_ridge"]
    assert set(actual) == set(expected)
    for key in ("transform", "adapted_transform", "adapted_fit"):
        np.testing.assert_array_equal(actual[key][:, 1:], transformed[:, 1:])
    names = [line.split("|")[1] for line in output.splitlines() if line.startswith("name|")]
    spans = {
        name: slice(int(start), int(stop))
        for _, name, start, stop in (
            line.split("|") for line in output.splitlines() if line.startswith("span|")
        )
    }
    assert names == columns.get_feature_names_out().tolist()
    assert spans == columns.output_indices_
    # Float64 reductions and the Cholesky solve can round differently.
    errors = {key: io.close(actual[key], value, 1e-12, 1e-12, key) for key, value in expected.items()}
    print(f"mixed Ridge pipeline: one-hot values, names and spans exact, max numeric error {max(errors.values()):.3g}")


if __name__ == "__main__":
    main()
