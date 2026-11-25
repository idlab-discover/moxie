"""Compare native named GaussianNB with pandas input to pinned scikit-learn."""

import inspect
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import validation

REFERENCE = Path(__file__).resolve().parents[4] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
assert subprocess.check_output(
    ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True,
).strip() == REVISION
for obj, path in (
    (GaussianNB, "sklearn/naive_bayes.py"),
    (validation, "sklearn/utils/validation.py"),
):
    pinned = subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{REVISION}:{path}"])
    assert Path(inspect.getfile(obj)).read_bytes() == pinned, path

actual_names, actual_predictions, actual_values, actual_scores = [], [], [], []
for line in subprocess.check_output([sys.argv[1]], text=True).splitlines():
    match line.split("|"):
        case ["name", name]:
            actual_names.append(name)
        case ["prediction", label, *values]:
            actual_predictions.append(int(label))
            actual_values.append([float(value) for value in values])
        case ["score", score]:
            actual_scores.append(float(score))
        case _:
            raise ValueError(line)

X = np.array([[1.0, 3.0], [3.0, 2.0], [2.0, 4.0], [4.0, 1.0]])
table = pd.DataFrame(X, columns=["height", "café"])
y = [0, 1, 0, 1]
weights = [1.0, 2.0, 2.0, 1.0]
model = GaussianNB().fit(table, y, sample_weight=weights)
for invalid in (
    table[["café", "height"]], table.drop(columns="café"), table.rename(columns={"café": "width"}),
):
    for method in (model.predict, model.predict_proba, lambda X: model.partial_fit(X, y)):
        try:
            method(invalid)
        except ValueError as error:
            assert "feature names" in str(error)
        else:
            raise AssertionError("inconsistent names were accepted")

model.partial_fit(table, y, sample_weight=2.0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.partial_fit(X, y, sample_weight=weights)
assert any("does not have valid feature names" in str(record.message) for record in caught)
assert actual_names == model.feature_names_in_.tolist()
np.testing.assert_array_equal(actual_predictions, model.predict(table))
np.testing.assert_allclose(
    actual_values,
    np.column_stack(
        (model.predict_proba(table), model.predict_log_proba(table), model.predict_joint_log_proba(table))
    ),
    rtol=1e-12, atol=1e-12,
)
np.testing.assert_allclose(actual_scores, [model.score(table, [0, 0, 0, 1], sample_weight=weights)],
                           rtol=0, atol=1e-15)

model.fit(X, y)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.partial_fit(table, y)
assert any("fitted without feature names" in str(record.message) for record in caught)
assert not hasattr(model, "feature_names_in_")
print("PASS GaussianNB named feature validation, lifecycle, probabilities and weighted score")
