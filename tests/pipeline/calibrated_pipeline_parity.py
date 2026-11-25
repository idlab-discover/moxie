"""Compare whole-pipeline sigmoid calibration with pinned scikit-learn."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

sources = {
    "pipeline": "sklearn/pipeline.py",
    "scalers": "sklearn/preprocessing/_data.py",
    "svc_base": "sklearn/svm/_base.py",
    "svc": "sklearn/svm/_classes.py",
    "calibration": "sklearn/calibration.py",
}
io.verify_reference(sources.values())
pipeline = io.load_pinned(sources["pipeline"])
scalers = io.load_pinned(sources["scalers"])
io.load_pinned(sources["svc_base"], register=True)
svc = io.load_pinned(sources["svc"])
calibration = io.load_pinned(sources["calibration"])
io.pinned_code(pipeline.Pipeline.fit, sources["pipeline"])
io.pinned_code(scalers.StandardScaler.fit, sources["scalers"])
io.pinned_code(svc.SVC.fit, sources["svc_base"])
io.pinned_code(svc.SVC.decision_function, sources["svc_base"])
io.pinned_code(calibration.CalibratedClassifierCV.fit, sources["calibration"])

binary = ROOT / "builds/tests/pipeline/test_calibrated_pipeline"
actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)
assert set(actual) == {"ensemble", "pooled"}

i = np.arange(36)
label = i % 2
X = np.column_stack((1.5 * label + (7 * i) % 11 / 4, (5 * i) % 13 / 2 - label))
y = np.where(label == 0, 9, -3).astype(np.float64)
i = np.arange(8)
query = np.column_stack((i / 2, (7 * i) % 9 / 2))

for name, native in actual.items():
    estimator = pipeline.make_pipeline(scalers.StandardScaler(), svc.SVC(tol=1e-8))
    model = calibration.CalibratedClassifierCV(estimator, cv=3, ensemble=name == "ensemble")
    model.fit(X, y)
    means = [pair.estimator.named_steps.standardscaler.mean_
             for pair in model.calibrated_classifiers_]
    io.close(native["means"], means, 0, 1e-14, f"{name} fitted scaler means")
    error = io.close(native["probabilities"], model.predict_proba(query),
                     0, 1e-7, f"{name} probabilities")
    io.close(native["predictions"], model.predict(query), 0, 0, f"{name} predictions")
    model.fit(X, y + 2)
    io.close(native["refit_classes"], model.classes_, 0, 0, f"{name} refit classes")
    print(f"PASS: {name} calibration, exact predictions and refit classes; max probability error {error:.3g}")
