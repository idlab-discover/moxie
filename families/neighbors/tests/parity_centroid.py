"""Compare native results with pinned sklearn centroid and mixin method bodies.

Only dense array validation, label encoding and pairwise distances are adapted;
training, shrinkage, priors and probability logic execute the original source.
Python is validation tooling only. This is not sklearn's full estimator suite.
"""
import ast
import itertools
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SOURCES = ["sklearn/neighbors/_nearest_centroid.py", "sklearn/discriminant_analysis.py"]
assert subprocess.check_output(
    ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
).strip() == REVISION
assert not subprocess.check_output(
    ["git", "-C", str(REFERENCE), "diff", "HEAD", "--", *SOURCES], text=True
)


def validate_data(model, X, y=None, *, reset=True, **options):
    X = np.array(X, dtype=options.get("dtype"), copy=True)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("invalid dense features")
    if reset:
        model.n_features_in_ = X.shape[1]
    elif X.shape[1] != model.n_features_in_:
        raise ValueError("feature mismatch")
    return X if y is None else (X, np.asarray(y).ravel())


class LabelEncoder:
    def fit_transform(self, y):
        self.classes_, inverse = np.unique(y, return_inverse=True)
        return inverse


def distances(X, Y, metric):
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("nonfinite pairwise input")
    return cdist(X, Y, "cityblock" if metric == "manhattan" else metric)


namespace = dict(
    np=np, sp=sparse, warnings=warnings, BaseEstimator=type("BaseEstimator", (), {}),
    ClassifierMixin=type("ClassifierMixin", (), {}), LabelEncoder=LabelEncoder,
    validate_data=validate_data, check_is_fitted=lambda *a: None,
    check_classification_targets=lambda y: None,
    pairwise_distances=distances,
    pairwise_distances_argmin=lambda X, Y, metric: distances(X, Y, metric).argmin(axis=1),
    get_tags=lambda self: type("Tags", (), {"input_tags": type("Input", (), {"allow_nan": False})})(),
    available_if=lambda condition: lambda method: method,
)
for source, name in [(SOURCES[1], "DiscriminantAnalysisPredictionMixin"), (SOURCES[0], "NearestCentroid")]:
    path = REFERENCE / source
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if not (isinstance(n, ast.AnnAssign) or isinstance(n, ast.FunctionDef) and n.name == "__sklearn_tags__")]
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
ReferenceCentroid = namespace["NearestCentroid"]


def fixture(scenario, dtype):
    n, d, classes = 13, 3, 3
    if scenario == 0:
        n, d, classes = 6, 2, 2
    elif scenario == 1:
        n, d, classes = 5, 2, 2
    elif scenario == 3:
        n, d, classes = 2, 1, 2
    elif scenario == 4:
        n, d, classes = 4, 1, 2
    X, y = np.empty((n, d), dtype=dtype), np.empty(n, dtype=dtype)
    for i in range(n):
        c = (i*i+2*i+1) % classes if classes == 2 else (0 if i < 3 else 1 if i < 7 else 2)
        for j in range(d):
            value = ((i*7+j*3) % 17)/5 + c*2-j
            if scenario == 5 and j == 0:
                value = 4
            if scenario == 6 and j == 1:
                value *= 100
            if scenario == 7:
                value += (j+1)*123
            X[i, j] = value
        y[i] = c*7-3
    if scenario == 0:
        X[:] = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y[:] = [-1, -1, -1, 1, 1, 1]
    elif scenario == 1:
        X[:] = [[0, 1], [1, 0], [1, 1], [2, 0], [6, 8]]
        y[:] = [1, 1, 2, 2, 2]
    elif scenario in (3, 4):
        X[:, 0] = [0]*(n//2) + [2]*(n//2)
        y[:] = [0]*(n//2) + [1]*(n//2)
    query = np.array([[(i*3-j*2)/4 for j in range(d)] for i in range(7)], dtype=dtype)
    if scenario == 6:
        query[:, 1] *= 100
    if scenario == 7:
        query += np.arange(1, d+1)*123
    if scenario == 8:
        X, y = X[::-1], y[::-1]
    return X, y, query, classes


def compare(dtype_name, metric, threshold, prior_mode, scenario):
    dtype = np.float32 if dtype_name == "f32" else np.float64
    X, y, query, classes = fixture(scenario, dtype)
    priors = prior_mode
    if prior_mode in ("explicit", "zero"):
        priors = list(range(1, classes+1))
        if prior_mode == "zero":
            priors[0] = 0
    model = ReferenceCentroid(metric, shrink_threshold=threshold, priors=priors)
    model.fit(X, y)
    raw = subprocess.check_output([
        str(ROOT / "builds/test_centroid"), dtype_name, metric,
        "none" if threshold is None else str(threshold), prior_mode, str(scenario)
    ], text=True)
    got = {}
    for line in raw.splitlines():
        parts = line.split()
        got.setdefault(parts[0], []).extend(float(v) for v in parts[1:])
    tol = 3e-5 if dtype_name == "f32" else 2e-11
    for key, expected in [
        ("C", model.classes_), ("M", model.centroids_), ("V", model.deviations_),
        ("R", model.class_prior_), ("S", model.within_class_std_dev_),
    ]:
        np.testing.assert_allclose(got[key], expected.ravel(), rtol=tol, atol=tol, equal_nan=True,
                                   err_msg=str((dtype_name, metric, threshold, prior_mode, scenario, key)))
    try:
        expected = model.predict(query)
    except ValueError:
        assert "P_ERROR" in got
    else:
        np.testing.assert_array_equal(got["P"], expected)
    if metric == "euclidean":
        try:
            expected = [model.decision_function(query), model.predict_proba(query), model.predict_log_proba(query)]
        except ValueError:
            assert "D_ERROR" in got
        else:
            for key, values in zip(["D", "B", "L"], expected):
                response_tol = 2e-9 if dtype_name == "f64" else tol
                np.testing.assert_allclose(got[key], values.ravel(), rtol=response_tol, atol=response_tol, equal_nan=True,
                                           err_msg=str((dtype_name, metric, threshold, prior_mode, scenario, key)))
    if scenario == 1 and metric == "euclidean" and threshold == 0.1:
        # Independent published R/pamr fixture retained by sklearn's test suite.
        np.testing.assert_allclose(np.array(got["M"]).reshape(2, 2),
                                   [[0.7787310, 0.8545292], [2.814179, 2.763647]], atol=1e-6)


with warnings.catch_warnings(), np.errstate(all="ignore"):
    warnings.simplefilter("ignore")
    cases = 0
    for settings in itertools.product(["f32", "f64"], ["euclidean", "manhattan"],
                                       [None, 0.1, 10.0], ["uniform", "empirical", "explicit", "zero"], range(9)):
        compare(*settings)
        cases += 1
print(f"PASS NearestCentroid: {cases} pinned-source configurations, learned state, labels, decision/probability outputs and pamr fixture")
