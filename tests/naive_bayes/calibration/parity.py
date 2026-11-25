"""Compare native weighted cv=2 Naive Bayes calibration with pinned sklearn."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

NB = "sklearn/naive_bayes.py"
CALIBRATION = "sklearn/calibration.py"
FROZEN = "sklearn/frozen/_frozen.py"
io.verify_reference([NB, CALIBRATION, FROZEN])
nb = io.load_pinned(NB)
frozen_module = io.load_pinned(FROZEN)
calibration = io.load_pinned(CALIBRATION)
# Both modules and the wrapper under test must use the same class identity.
calibration.FrozenEstimator = frozen_module.FrozenEstimator
io.pinned_code(calibration.CalibratedClassifierCV.fit, CALIBRATION)
io.pinned_code(frozen_module.FrozenEstimator.fit, FROZEN)

rng = np.random.default_rng(1137)


def samples(size):
    label = np.arange(size) % 2
    X = rng.integers(0, 4, size=(size, 4))
    X[:, 0] = (rng.integers(0, 4, size=size) + label) // 2
    X[:, 1] = (X[:, 1] + label) // 2
    return X.astype(np.float64), np.where(label == 0, 9.0, -3.0)


train, train_y = samples(100)
X, y = samples(80)
query, _ = samples(24)
weights = (1 + np.arange(y.size) % 7) / 4
binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(dir=binary.parent) as temporary:
    fixture = Path(temporary) / "calibration.txt"
    fixture.write_text(io.case_line("weighted", 64, {}, {
        "train": train, "train_y": train_y, "X": X, "y": y,
        "query": query, "weights": weights,
    }) + "\n")
    result = subprocess.run([binary, fixture], check=True, text=True, capture_output=True)
actual = io.parse_results(result.stdout)

for name in ("GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB"):
    model_type = getattr(nb, name)
    io.pinned_code(model_type.fit, NB)
    frozen_base = model_type().fit(train, train_y)
    for method in ("sigmoid", "isotonic"):
        for frozen in (False, True):
            case = f"{name}{'/frozen' if frozen else ''}/{method}"
            estimator = frozen_module.FrozenEstimator(frozen_base) if frozen else model_type()
            model = calibration.CalibratedClassifierCV(estimator, method=method, cv=2)
            model.fit(X, y, sample_weight=weights)
            cases = [case]
            if frozen and name == "GaussianNB":
                cases.append(f"{name}/frozen_adapter/{method}")
            for ident in cases:
                native = actual.pop(ident)
                io.close(native["classes"], model.classes_, 0, 0, ident)
                io.close(native["predictions"], model.predict(query), 0, 0, ident)
                # Sigmoid convergence permits small optimizer differences.
                tolerance = 1e-7 if method == "sigmoid" else 1e-12
                error = io.close(
                    native["probabilities"], model.predict_proba(query), 0, tolerance, ident,
                )
                if frozen:
                    io.close(
                        native["base_probabilities"], frozen_base.predict_proba(query),
                        0, 1e-12, ident,
                    )
                print(f"PASS {ident} (max calibrated probability error {error:.3g})")
assert not actual, actual.keys()
