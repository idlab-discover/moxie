"""Compare scaler plus ClassicalMDS training embeddings with pinned sklearn."""
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
MDS = "sklearn/manifold/_classical_mds.py"
hashes = io.verify_reference([PIPELINE, PREPROCESSING, MDS])
pipeline = io.load_pinned(PIPELINE)
scalers = io.load_pinned(PREPROCESSING)
manifold = io.load_pinned(MDS)
io.pinned_code(pipeline.Pipeline.fit_transform, PIPELINE)
io.pinned_code(scalers.StandardScaler.fit, PREPROCESSING)
io.pinned_code(manifold.ClassicalMDS.fit_transform, MDS)

actual = io.parse_results(io.run(ROOT, BUILD / "test_pipeline_embedding", capture=True).stdout)
assert set(actual) == {"float32", "float64"}
errors = {}
for dtype, tolerance in [(np.float32, 2e-6), (np.float64, 2e-11)]:
    X = np.array([[0, 3, 1], [1, 5, 5], [2, 4, 2], [3, 0, 6], [5, 1, 4]], dtype=dtype)
    pipe = pipeline.make_pipeline(scalers.StandardScaler(), manifold.ClassicalMDS())
    assert hasattr(pipe, "fit_transform")
    assert not hasattr(pipe, "transform")
    assert not hasattr(pipe, "inverse_transform")
    fitted = pipe.fit_transform(X)
    fresh = pipeline.clone(pipe).fit(X)
    refit = pipe.set_params(classicalmds__n_components=1).fit_transform(X, np.zeros(len(X)))
    nested = pipeline.make_pipeline(scalers.StandardScaler(), pipeline.clone(fresh))
    expected = {
        "fitted": fitted,
        "fit": fresh[-1].embedding_,
        "refit": refit,
        "nested": nested.fit_transform(X),
    }
    case = str(X.dtype)
    assert set(actual[case]) == set(expected)
    # Float32 scaling and pairwise-distance rounding can differ before the
    # Float64 eigensolve. Float64 inputs primarily differ in the eigensolver.
    errors[case] = {
        name: io.close(actual[case][name], value, tolerance, tolerance, f"{case} {name}")
        for name, value in expected.items()
    }

(BUILD / "embedding-parity.json").write_text(json.dumps({
    "reference": io.REVISION,
    "source_sha256": hashes,
    "max_abs_errors": errors,
}, indent=2) + "\n")
print("PASS: pinned sklearn StandardScaler → ClassicalMDS fit, fit_transform, refit and nesting, Float32 and Float64")
