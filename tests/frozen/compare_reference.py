"""Compare pure Mojo results with pinned local sklearn source modules.

Run with a Python environment containing sklearn and NumPy; see docs/frozen/api.md.
Python is reference tooling only, never part of the Mojo library runtime.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn-root", type=Path, default=ROOT.parent / "scikit-learn")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.sklearn_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected sklearn {REVISION}, got {revision}")
    subprocess.run(
        ["git", "-C", str(args.sklearn_root), "diff", "--exit-code", "HEAD", "--",
         "sklearn/dummy.py", "sklearn/frozen/_frozen.py"], check=True,
    )
    DummyRegressor = load(
        "sklearn._moxie_reference_dummy", args.sklearn_root / "sklearn/dummy.py"
    ).DummyRegressor
    FrozenEstimator = load(
        "sklearn._moxie_reference_frozen", args.sklearn_root / "sklearn/frozen/_frozen.py"
    ).FrozenEstimator
    assert Path(FrozenEstimator.fit.__code__.co_filename).resolve() == (args.sklearn_root / "sklearn/frozen/_frozen.py").resolve()
    assert Path(DummyRegressor.fit.__wrapped__.__code__.co_filename).resolve() == (args.sklearn_root / "sklearn/dummy.py").resolve()
    output = subprocess.check_output(
        [str(ROOT / "builds/tests/frozen/reference_frozen")],
        cwd=ROOT, text=True,
    )
    actual = {}
    for line in output.splitlines():
        case, name, value = line.split(maxsplit=2)
        key = (int(case), name)
        if key in actual:
            raise AssertionError(f"Duplicate observation: {key}")
        actual[key] = json.loads(value)

    expected = {}
    X = np.arange(4.0)
    changed = np.array([100.0, 101.0, 103.0, 108.0])
    for case in range(5):
        y = (case - 2) * 8.0 + np.array([0.0, 1.0, 3.0, 8.0])
        frozen = FrozenEstimator(DummyRegressor().fit(X, y))
        expected[case, "before"] = frozen.predict(X)
        expected[case, "score"] = frozen.score(X, y)
        for _ in range(3):
            frozen.fit(X, changed)
        frozen.fit([], changed)
        frozen.fit("ignored", 42)
        cloned = clone(frozen)
        assert cloned is frozen  # Python identity; native clones share inner state.
        expected[case, "after"] = frozen.predict(X)
        expected[case, "clone"] = cloned.predict(X)
        expected[case, "empty"] = frozen.predict([])
        expected[case, "changed_score"] = frozen.score(X, changed)
        params = frozen.get_params()
        assert list(params) == ["estimator"]
        assert frozen.get_params(False)["estimator"] is params["estimator"]
        try:
            frozen.set_params(estimator__strategy="median")
        except ValueError:
            pass
        else:
            raise AssertionError("Reference accepted nested mutation")
        frozen.set_params(estimator=DummyRegressor().fit(X, changed))
        expected[case, "replaced"] = frozen.predict(X)
        frozen.set_params(**params)
        expected[case, "restored"] = frozen.predict(X)
    try:
        FrozenEstimator(DummyRegressor()).fit(X, changed)
    except NotFittedError:
        pass
    else:
        raise AssertionError("Reference trained an unfitted frozen estimator")

    # Mirror only the native dispatch probes; these are not estimator parity.
    class ResponseProbe(sklearn.base.BaseEstimator):
        def __sklearn_is_fitted__(self): return True
        def fit(self, X, y=None, **kwargs): raise AssertionError("inner fit called")
        def predict(self, X): return X + 10
        def predict_proba(self, X): return [0.25, 0.75]
        def predict_log_proba(self, X): return [-1.3862943611198906, -0.2876820724517809]
        def decision_function(self, X): return X - 10.0
        def score(self, X, y, sample_weight=None): return float(X + y + (sample_weight or 0))
    response = FrozenEstimator(ResponseProbe())
    response.fit(1, 2, sample_weight=-100)
    expected[5, "predict"] = clone(response).predict(2)
    expected[5, "proba"] = response.predict_proba(3)
    expected[5, "log_proba"] = response.predict_log_proba(3)
    expected[5, "decision"] = response.decision_function(3)
    expected[5, "score"] = response.score(3, 4)
    expected[5, "weighted_score"] = response.score(3, 4, sample_weight=5)
    class TransformProbe(ResponseProbe):
        def transform(self, X): return X + 7
        def inverse_transform(self, X): return X - 7
    transform = FrozenEstimator(TransformProbe())
    transform.fit(100, None)
    expected[6, "transform"] = transform.transform(3)
    expected[6, "inverse"] = transform.inverse_transform(10)
    class DensityProbe(ResponseProbe):
        def score(self, X, y=None): return float(X - 2 + (len(y) if y else 0))
        def score_samples(self, X): return [float(X), -2.0]
    density = FrozenEstimator(DensityProbe())
    density.fit(100, None)
    expected[7, "score"] = density.score(5)
    expected[7, "target_score"] = density.score(5, "abc")
    expected[7, "samples"] = density.score_samples(5)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        # Predictions and these exactly representable means must match exactly.
        # R2 may differ in reduction order: permit only Float64 rounding error.
        tolerance = 1e-12 if "score" in key[1] else 0
        assert np.shape(actual[key]) == np.shape(value), key
        np.testing.assert_allclose(actual[key], value, rtol=tolerance, atol=tolerance, err_msg=str(key))
    print(f"Passed {len(expected)} native/reference comparisons at sklearn {REVISION}")
    print(f"Reference support dependencies: sklearn={sklearn.__version__}, numpy={np.__version__}, scipy={scipy.__version__}")


if __name__ == "__main__":
    main()
