"""Exact comparisons for typed numeric categorical imputation and composition."""
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn.compose
import sklearn.impute

SOURCE = Path(__file__).resolve()
sys.path.insert(0, str(SOURCE.parents[1]))
sys.path.insert(0, str(SOURCE.parents[6] / "tools"))
from oracle import load
import compose_parity_io as io


def compare_matrix(records, expected, parse):
    shape = tuple(map(int, next(records).split()))
    assert shape == expected.shape
    # Compare Python scalars to avoid NumPy coercing adjacent large integers.
    assert [parse(next(records)) for _ in range(expected.size)] == expected.ravel().tolist()


def main(binary):
    imputer = load().SimpleImputer
    records = iter(subprocess.check_output([binary], text=True).splitlines())
    names = ["count", "empty", "complete"]
    for parse, (lo, hi, extra) in [
        (int, (2**53, 2**53 + 1, 2**53 + 2)),
        (float, (-1.5, 0.5, 9.25)),
        (float, (-1.5, 0.5, 9.25)),
    ]:
        # Object cells represent nullable categorical values without promoting
        # Int64 to float when NaN marks a missing cell.
        train = np.array([
            [hi, np.nan, hi], [np.nan, np.nan, hi],
            [lo, np.nan, lo], [np.nan, np.nan, hi],
        ], dtype=object)
        query = np.array([[hi, hi, extra], [np.nan, np.nan, np.nan]], dtype=object)
        for strategy in ("most_frequent", "constant"):
            for keep in (False, True):
                model = imputer(strategy=strategy, fill_value=extra, keep_empty_features=keep)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                    fitted = model.fit_transform(train)
                    transformed = model.transform(query)
                for expected in model.statistics_:
                    actual = next(records)
                    if np.isnan(expected):
                        assert actual == "missing"
                    else:
                        assert parse(actual) == expected
                name_count = int(next(records))
                assert [next(records) for _ in range(name_count)] == model.get_feature_names_out(names).tolist()
                compare_matrix(records, fitted, parse)
                compare_matrix(records, transformed, parse)

    boolean_train = np.array([[False, np.nan], [True, np.nan], [np.nan, np.nan]], dtype=object)
    for strategy in ("most_frequent", "constant"):
        model = imputer(strategy=strategy, fill_value=True, keep_empty_features=True)
        # The typed Boolean domain represents the retained-empty zero as False.
        compare_matrix(records, model.fit_transform(boolean_train).astype(bool), int)

    encoder_path = "sklearn/preprocessing/_encoders.py"
    encode_path = "sklearn/utils/_encode.py"
    pipeline_path = "sklearn/pipeline.py"
    compose_path = "sklearn/compose/_column_transformer.py"
    io.verify_reference([encoder_path, encode_path, pipeline_path, compose_path])
    io.load_pinned(encode_path, register=True)
    pipeline = io.load_pinned(pipeline_path, register=True)
    compose = io.load_pinned(compose_path)
    encoder = io.load_pinned(encoder_path).OrdinalEncoder
    io.pinned_code(encoder.fit, encoder_path)
    io.pinned_code(compose.ColumnTransformer.fit_transform, compose_path)
    io.pinned_code(pipeline.Pipeline.fit_transform, pipeline_path)
    import pandas as pd

    train = pd.DataFrame({"count": pd.Series([2**53 + 1, np.nan, 2**53], dtype=object)})
    query = pd.DataFrame({"count": pd.Series([np.nan, 2**53 + 1], dtype=object)})
    model = compose.ColumnTransformer([
        ("category", pipeline.make_pipeline(imputer(strategy="most_frequent"), encoder()), ["count"]),
    ])
    compare_matrix(records, model.fit_transform(train), float)
    compare_matrix(records, model.transform(query), float)
    assert [next(records)] == model.get_feature_names_out().tolist()
    assert next(records, None) is None
    print("PASS: 14 categorical imputation cases and exact ColumnTransformer pipeline parity")


if __name__ == "__main__":
    main(sys.argv[1])
