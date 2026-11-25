"""Compare inverse transforms from transformer-only pipelines with pinned sklearn."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "builds/tests/pipeline"
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

PIPELINE = "sklearn/pipeline.py"
PREPROCESSING = "sklearn/preprocessing/_data.py"
hashes = io.verify_reference([PIPELINE, PREPROCESSING])
pipeline = io.load_pinned(PIPELINE)
scalers = io.load_pinned(PREPROCESSING)
io.pinned_code(pipeline.Pipeline.inverse_transform, PIPELINE)
io.pinned_code(scalers.StandardScaler.inverse_transform, PREPROCESSING)
io.pinned_code(scalers.MinMaxScaler.inverse_transform, PREPROCESSING)

output = io.run(ROOT, BUILD / "test_transforming_pipeline", capture=True).stdout
actual = io.parse_results(output)
assert set(actual) == {"float32", "float64"}
errors = {}
for dtype, tolerance in [(np.float32, 2e-6), (np.float64, 1e-12)]:
    X = np.array([[2, 3], [4, 9], [6, 3]], dtype=dtype)
    pipe = pipeline.make_pipeline(scalers.StandardScaler(), scalers.MinMaxScaler())
    transformed = pipe.fit_transform(X)
    restored = pipe.inverse_transform(transformed)
    assert restored.dtype == dtype
    fresh = pipeline.clone(pipe).set_params(minmaxscaler__feature_range=(-1, 1)).fit(X)
    expected = {
        "restored": restored,
        "new_values": pipe.inverse_transform(np.zeros((2, 2), dtype=dtype)),
        "refit": fresh.inverse_transform(transformed),
        "original": pipe.inverse_transform(transformed),
    }
    case = str(X.dtype)
    assert set(actual[case]) == set(expected)
    errors[case] = {
        name: io.close(actual[case][name], value, tolerance, tolerance, f"{case} {name}")
        for name, value in expected.items()
    }

(BUILD / "transforming-parity.json").write_text(json.dumps({
    "reference": io.REVISION,
    "source_sha256": hashes,
    "max_abs_errors": errors,
}, indent=2) + "\n")
print("PASS: pinned sklearn inverse transforms before and after refit, Float32 and Float64")
