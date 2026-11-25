"""Compare one mixed table with the pinned sklearn composition.

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

train = pd.DataFrame({
    "color": pd.Categorical(["red", "blue", "red", "amber", "blue"]),
    "age": [10.0, np.nan, 30.0, 40.0, 20.0],
})
test = pd.DataFrame({
    "age": [35.0, np.nan, 15.0],
    "color": pd.Categorical(["green", None, "red"]),
})
model = compose.ColumnTransformer([
    ("num", pipeline.make_pipeline(impute.SimpleImputer(strategy="median"), data.StandardScaler()),
     compose.make_column_selector(dtype_include="number")),
    ("cat", encoders.OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     compose.make_column_selector(dtype_include="category")),
])
expected_fit = model.fit_transform(train)
expected_transform = model.transform(test)
output = io.run(ROOT, BUILD / "mixed", capture=True).stdout
actual = io.parse_results(output)["mixed"]
errors = {}
for label, expected in {
    "fit": expected_fit,
    "transform": expected_transform,
    "after_failed_refit": expected_transform,
}.items():
    errors[label] = io.close(actual[label][:, :1], expected[:, :1], 1e-12, 1e-12, label)
    np.testing.assert_array_equal(actual[label][:, 1:], expected[:, 1:])
names = [line.split("|")[1] for line in output.splitlines() if line.startswith("name|")]
spans = {name: slice(int(start), int(stop)) for _, name, start, stop in
         (line.split("|") for line in output.splitlines() if line.startswith("span|"))}
assert names == model.get_feature_names_out().tolist(), names
assert spans == model.output_indices_, spans

report = BUILD / "parity"
report.mkdir(parents=True, exist_ok=True)
(report / "mixed-report.json").write_text(json.dumps({
    "revision": io.REVISION, "sources": hashes, "scaler_max_errors": errors,
    "names": names, "spans": {name: [span.start, span.stop] for name, span in spans.items()},
}, indent=2) + "\n")
print("Mixed table parity passed: one-hot values, names and spans exact, scaler within 1e-12")
