"""Compare Ridge target-transformation CV, grid search and refit with pinned sklearn."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    target_source = "sklearn/compose/_target.py"
    function_source = "sklearn/preprocessing/_function_transformer.py"
    ridge_base = "sklearn/linear_model/_base.py"
    ridge_source = "sklearn/linear_model/_ridge.py"
    split_source = "sklearn/model_selection/_split.py"
    validation_source = "sklearn/model_selection/_validation.py"
    search_source = "sklearn/model_selection/_search.py"
    indexing_source = "sklearn/utils/metaestimators.py"
    io.verify_reference([
        target_source, function_source, ridge_base, ridge_source,
        split_source, validation_source, search_source, indexing_source,
    ])
    import sklearn.preprocessing

    function = io.load_pinned(function_source, register=True)
    sklearn.preprocessing.FunctionTransformer = function.FunctionTransformer
    target = io.load_pinned(target_source)
    io.load_pinned(ridge_base, register=True)
    ridge = io.load_pinned(ridge_source)
    indexing = io.load_pinned(indexing_source)
    split = io.load_pinned(split_source, register=True)
    validation = io.load_pinned(validation_source, register=True)
    validation._safe_split = indexing._safe_split
    search = io.load_pinned(search_source)

    for method, source in [
        (target.TransformedTargetRegressor.fit, target_source),
        (target.TransformedTargetRegressor.predict, target_source),
        (target.TransformedTargetRegressor.n_features_in_.fget, target_source),
        (function.FunctionTransformer.fit, function_source),
        (ridge.Ridge.fit, ridge_source),
        (split.KFold._iter_test_indices, split_source),
        (validation.cross_val_score, validation_source),
        (validation._fit_and_score, validation_source),
        (validation._safe_split, indexing_source),
        (search.GridSearchCV.fit, search_source),
    ]:
        io.pinned_code(method, source)
    assert target.FunctionTransformer is function.FunctionTransformer
    assert search._fit_and_score is validation._fit_and_score

    X = np.column_stack([np.arange(9, dtype=np.float64), [2, -1, 3, -2, 2, -3, 1, 0, 4]])
    y = np.array([9, 16, 25, 49, 64, 81, 121, 144, 196], dtype=np.float64)
    test = np.array([[1.5, 1], [4.5, -1], [8.5, 2]])
    model = target.TransformedTargetRegressor(
        regressor=ridge.Ridge(), func=np.sqrt, inverse_func=np.square,
    )
    cv = split.KFold(n_splits=3)
    scores = validation.cross_val_score(model, X, y, cv=cv, error_score="raise")
    grid = search.GridSearchCV(
        model, {"regressor__alpha": [25.0, 0.1, 1.0]}, cv=cv, error_score="raise",
    ).fit(X, y)
    expected = {
        "cv": scores,
        "candidate_scores": np.column_stack([
            grid.cv_results_[f"split{i}_test_score"] for i in range(3)
        ]),
        "best_index": grid.best_index_,
        "best_alpha": grid.best_params_["regressor__alpha"],
        "best_score": grid.best_score_,
        "n_features": grid.n_features_in_,
        "predict": grid.predict(test),
    }

    binary = ROOT / "builds/tests/compose/target/ridge_search"
    actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)["ridge_target_search"]
    assert set(actual) == set(expected)
    for name in ("best_index", "best_alpha", "n_features"):
        np.testing.assert_array_equal(actual[name], expected[name])
    # Allow Float64 rounding in the Ridge solve, inverse transform and R² reductions.
    errors = [
        io.close(actual[name], expected[name], 1e-12, 1e-12, name)
        for name in ("cv", "candidate_scores", "best_score", "predict")
    ]
    print(
        f"TransformedTargetRegressor Ridge search: best index {grid.best_index_}, "
        f"alpha {grid.best_params_['regressor__alpha']}, max absolute error {max(errors):.3g}"
    )


if __name__ == "__main__":
    main()
