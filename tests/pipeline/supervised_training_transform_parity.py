"""Compare supervised training-only final dispatch against pinned sklearn."""
from pathlib import Path
import sys

from sklearn.base import BaseEstimator, clone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io

PIPELINE = "sklearn/pipeline.py"
io.verify_reference([PIPELINE])
pipeline = io.load_pinned(PIPELINE)
io.pinned_code(pipeline.Pipeline.fit_transform, PIPELINE)


class Shift(BaseEstimator):
    def __init__(self, offset=2):
        self.offset = offset

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X + self.offset


class TrainingOnly(BaseEstimator):
    def fit(self, X, y):
        raise AssertionError("fit_transform must be preferred")

    def fit_transform(self, X, y=None):
        self.fitted_ = True
        return X + y + 100

    def predict(self, X):
        return X + 10


def pair(prefix, final):
    return pipeline.Pipeline([("prefix", prefix), ("final", final)])


pipe = pair(Shift(), TrainingOnly())
assert hasattr(pipe, "fit_transform")
assert not hasattr(pipe, "transform")
fitted = pipe.fit_transform(5, 7)
predicted = pipe.predict(1)
refitted = pipe.fit_transform(8, 3)
fresh = clone(pipe).set_params(prefix__offset=4)
nested = pair(Shift(3), fresh)
assert hasattr(nested, "fit_transform")
assert not hasattr(nested, "transform")
expected = [fitted, predicted, refitted, nested.fit_transform(5, 7), nested.predict(1)]

output = io.run(ROOT, ROOT / "builds/tests/pipeline/test_sequential", capture=True).stdout
actual, = [line.split()[1:] for line in output.splitlines() if line.startswith("training_only ")]
assert list(map(int, actual)) == expected
print(f"PASS: pinned sklearn supervised training-only fit_transform, predict, refit and nesting ({expected})")
