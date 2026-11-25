"""Compare final-step weight routing with pinned scikit-learn."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

PIPELINE = "sklearn/pipeline.py"
DUMMY = "sklearn/dummy.py"
PREPROCESSING = "sklearn/preprocessing/_data.py"
io.verify_reference([PIPELINE, DUMMY, PREPROCESSING])
pipeline = io.load_pinned(PIPELINE)
dummy = io.load_pinned(DUMMY)
scalers = io.load_pinned(PREPROCESSING)
io.pinned_code(pipeline.Pipeline.fit, PIPELINE)
io.pinned_code(dummy.DummyRegressor.fit, DUMMY)
io.pinned_code(scalers.StandardScaler.fit, PREPROCESSING)

binary = ROOT / "builds/tests/pipeline/test_pipeline_weighted_fit"
actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)
assert set(actual) == {"float32", "float64"}
for dtype, tolerance in [(np.float32, 2e-6), (np.float64, 1e-12)]:
    y = np.array([0, 2, 4, 6], dtype=dtype)
    X = y.reshape(-1, 1).copy()
    weights = np.array([1, 1, 1, 5], dtype=dtype)
    pipe = pipeline.Pipeline([("scale", scalers.StandardScaler()), ("model", dummy.DummyRegressor())])
    pipe.fit(X, y, model__sample_weight=weights)
    expected = actual[str(y.dtype)]
    io.close(expected["weighted"], pipe.predict(X), 0, 0, "weighted predictions")
    io.close(expected["scaled"], pipe.named_steps.scale.transform(X),
             tolerance, tolerance, "unweighted scaler")

    pipe.fit(X, y)
    io.close(expected["unweighted"], pipe.predict(X), 0, 0, "unweighted refit")
    nested = pipeline.Pipeline([
        ("copy", scalers.StandardScaler(with_mean=False, with_std=False)),
        ("tail", pipeline.Pipeline([
            ("scaler", scalers.StandardScaler()),
            ("regressor", dummy.DummyRegressor()),
        ])),
    ])
    nested.fit(X, y, tail__regressor__sample_weight=weights[::-1])
    io.close(expected["nested"], nested.predict(X), 0, 0, "nested weighted fit")
    for estimator in (nested, pipe):
        key = "tail__regressor__sample_weight" if estimator is nested else "model__sample_weight"
        with np.testing.assert_raises(ValueError):
            estimator.fit(X, y, **{key: weights[:1]})
    with np.testing.assert_raises(ValueError):
        nested.fit(X, y, sample_weight=weights)
    with np.testing.assert_raises(KeyError):
        nested.fit(X, y, tail__model__sample_weight=weights)

print("PASS: pinned sklearn named weighted fit, unweighted refit, nested routing and invalid inputs, Float32 and Float64")
