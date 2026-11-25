"""Replay dense SelectKBest/SVC CV against the pinned scikit-learn sources."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

sources = {
    "feature_selection": "sklearn/feature_selection/_univariate_selection.py",
    "pipeline": "sklearn/pipeline.py",
    "split": "sklearn/model_selection/_split.py",
    "svc_base": "sklearn/svm/_base.py",
    "svc": "sklearn/svm/_classes.py",
}
io.verify_reference(sources.values())
selection = io.load_pinned(sources["feature_selection"])
pipeline = io.load_pinned(sources["pipeline"])
split = io.load_pinned(sources["split"])
io.load_pinned(sources["svc_base"], register=True)
svc = io.load_pinned(sources["svc"])
io.pinned_code(selection.SelectKBest.fit, sources["feature_selection"])
io.pinned_code(selection.f_classif, sources["feature_selection"])
io.pinned_code(pipeline.Pipeline.fit, sources["pipeline"])
io.pinned_code(split.StratifiedKFold._make_test_folds, sources["split"])
io.pinned_code(svc.SVC.fit, sources["svc_base"])

i = np.arange(40)
group = i % 2
X = np.column_stack((
    (i * 7) % 17 / 4 + group,
    (i * 11) % 13 / 4 + group * (i < 25),
    (i * 13) % 19 / 4 + 2 * group * (i >= 25),
    (i * 3) % 11 / 4,
))
y = np.where(group == 0, 9.0, -3.0)

expected_folds = []
expected_predictions = []
for train, test in split.StratifiedKFold().split(X, y):
    pipe = pipeline.make_pipeline(selection.SelectKBest(k=1), svc.SVC())
    pipe.fit(X[train], y[train].tolist())
    feature, = pipe.named_steps.selectkbest.get_support(indices=True)
    expected_folds.append((pipe.score(X[test], y[test]), feature))
    expected_predictions.extend(zip(test, pipe.predict(X[test])))

full = selection.SelectKBest(k=1).fit(X, y)
full_feature, = full.get_support(indices=True)
assert any(feature != full_feature for _, feature in expected_folds)
refit = full.set_params(k=2).fit(X, y)

binary = ROOT / "builds/tests/pipeline/test_select_k_best_cv"
native = defaultdict(list)
for line in io.run(ROOT, binary, capture=True).stdout.splitlines():
    key, *values = line.split()
    if key != "PASS:":
        native[key].append([float(value) for value in values])

np.testing.assert_array_equal(native["fold"], expected_folds)
np.testing.assert_array_equal(native["prediction"], expected_predictions)
np.testing.assert_array_equal(native["full_support"], [[full_feature]])
np.testing.assert_array_equal(native["refit_support"], refit.get_support(indices=True)[:, None])
print("PASS: exact fold indices, selected columns, predictions, scores and refit support")
