"""Compare mixed-table cross-validation and search against pinned scikit-learn."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "columns"))
from mixed_parity import ROOT, compose, data, encoders, impute, io, pipeline


def main():
    ridge_base = "sklearn/linear_model/_base.py"
    ridge_source = "sklearn/linear_model/_ridge.py"
    split_source = "sklearn/model_selection/_split.py"
    validation_source = "sklearn/model_selection/_validation.py"
    search_source = "sklearn/model_selection/_search.py"
    indexing_source = "sklearn/utils/metaestimators.py"
    io.verify_reference([
        ridge_base, ridge_source, split_source, validation_source, search_source, indexing_source,
    ])
    io.load_pinned(ridge_base, register=True)
    ridge = io.load_pinned(ridge_source)
    indexing = io.load_pinned(indexing_source)
    split = io.load_pinned(split_source, register=True)
    validation = io.load_pinned(validation_source, register=True)
    # Use the pinned row-indexing implementation in each fold.
    validation._safe_split = indexing._safe_split
    search = io.load_pinned(search_source)
    for method, source in [
        (ridge.Ridge.fit, ridge_source),
        (split.KFold._iter_test_indices, split_source),
        (validation.cross_val_score, validation_source),
        (validation._fit_and_score, validation_source),
        (validation._safe_split, indexing_source),
        (search.GridSearchCV.fit, search_source),
    ]:
        io.pinned_code(method, source)
    assert search._fit_and_score is validation._fit_and_score

    train = pd.DataFrame({
        "age": [10.0, np.nan, 30.0, 40.0, 20.0, np.nan, 65.0, 52.0, 79.0],
        "color": ["red", "blue", "green", "red", "amber", "blue", "red", "amber", "red"],
    })
    test = pd.DataFrame({
        "color": ["green", "blue", "new"],
        "age": [35.0, np.nan, 15.0],
    })
    y = np.array([2, -1, 5, 7, 0, -2, 12, 8, 14], dtype=np.float64)
    columns = compose.ColumnTransformer([
        ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()), ["age"]),
        ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["color"]),
    ])
    model = pipeline.make_pipeline(columns, ridge.Ridge())
    cv = split.KFold(n_splits=3)
    scores = validation.cross_val_score(model, train, y, cv=cv, error_score="raise")
    grid = search.GridSearchCV(
        model, {"ridge__alpha": [25.0, 1.0, 0.1]}, cv=cv, error_score="raise",
    ).fit(train, y)
    expected = {
        "cv": scores,
        "candidate_scores": np.column_stack([
            grid.cv_results_[f"split{i}_test_score"] for i in range(3)
        ]),
        "best_index": grid.best_index_,
        "best_alpha": grid.best_params_["ridge__alpha"],
        "best_score": grid.best_score_,
        "predict": grid.predict(test),
    }
    binary = ROOT / "builds/tests/compose/pipeline/mixed_ridge_search"
    output = io.run(ROOT, binary, capture=True).stdout
    actual = io.parse_results(output)["mixed_ridge_search"]
    assert set(actual) == set(expected)
    for name in ("best_index", "best_alpha"):
        np.testing.assert_array_equal(actual[name], expected[name])
    # Float64 reductions, the Cholesky solve, and fold averaging can round differently.
    errors = {name: io.close(actual[name], value, 1e-12, 1e-12, name)
              for name, value in expected.items()}
    print(f"mixed Ridge search: best index and alpha exact, max numeric error {max(errors.values()):.3g}")


if __name__ == "__main__":
    main()
