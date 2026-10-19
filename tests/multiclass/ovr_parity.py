"""Compare moxie OneVsRestClassifier with the pinned local scikit-learn.

Setup (from the repository root, once):
    uv venv builds/reference-env --python 3.12
    uv pip install --python builds/reference-env/bin/python ../scikit-learn
Run after `pixi run build`:
    builds/reference-env/bin/python tests/multiclass/ovr_parity.py

scikit-learn's own OneVsRestClassifier wraps Python twins of the Mojo test
components in tests/ovr_fixtures.mojo. Python only creates fixtures, launches
the native driver and compares results; the library runtime is pure Mojo.
"""
from pathlib import Path
import inspect
import json
import math
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn import multiclass

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
assert subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip() == REVISION
assert Path(inspect.getfile(multiclass)).read_bytes() == (REFERENCE / "sklearn/multiclass.py").read_bytes(), (
    "install the pinned local scikit-learn checkout into builds/reference-env")
OUT = ROOT / "builds" / "multiclass-parity"
OUT.mkdir(parents=True, exist_ok=True)

# Decision values use the same Float64 operation order in both languages; they
# differ only by FMA contraction in the native build (observed <= 4e-15 abs).
# Fixture probabilities also pass through exp: Mojo 1.0 std.math.exp differs
# from CPython's libm exp by up to ~1.2e-11 relative for |x| <= 30, growing with
# |x| (measured). OvR itself only copies, stacks and row-normalizes these
# values, so a 5e-11 relative bound isolates the OvR logic from fixture libm.
TOLERANCE = {"decision": (1e-12, 1e-14), "proba": (5e-11, 1e-14)}


class _Centroids(ClassifierMixin, BaseEstimator):
    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        assert set(np.unique(y)) <= {-1, 0, 1}
        self.classes_ = np.array([0, 1])
        p = X.shape[1]
        s0, s1, n0, n1 = [0.0] * p, [0.0] * p, 0, 0
        for i in range(X.shape[0]):
            for j in range(p):
                if y[i] == 1:
                    s1[j] += X[i, j]
                else:
                    s0[j] += X[i, j]
            n1 += y[i] == 1
            n0 += y[i] != 1
        self.mean0_ = [v / n0 for v in s0]
        self.mean1_ = [v / n1 for v in s1]
        self.n_features_in_ = p
        return self

    def _margin(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            d0 = d1 = 0.0
            for j in range(X.shape[1]):
                a = X[i, j] - self.mean0_[j]
                b = X[i, j] - self.mean1_[j]
                d0 += a * a
                d1 += b * b
            out[i] = (d0 - d1) + self.shift
        return out

    def _proba(self, X):
        p = np.array([1.0 / (1.0 + math.exp(-m)) if not math.isnan(m) else math.nan for m in self._margin(X)])
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self._margin(X) > 0).astype(int)


class MarginClassifier(_Centroids):
    def decision_function(self, X):
        return self._margin(X)


class ProbaClassifier(_Centroids):
    def predict_proba(self, X):
        return self._proba(X)


class MarginProbaClassifier(_Centroids):
    def decision_function(self, X):
        return self._margin(X)

    def predict_proba(self, X):
        return self._proba(X)


class ScoreRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        self.model_ = _Centroids(shift=self.shift).fit(X, y)
        return self

    def predict(self, X):
        return self.model_._margin(X)


class DecisionRegressor(ScoreRegressor):
    def decision_function(self, X):
        model = self.model_
        old = model.shift
        model.shift = self.shift + 10.0
        try:
            return model._margin(X)
        finally:
            model.shift = old


class Float32Regressor(ScoreRegressor):
    def predict(self, X):
        return super().predict(X).astype(np.float32)


COMPONENTS = {"MarginClassifier": MarginClassifier, "ProbaClassifier": ProbaClassifier,
              "MarginProbaClassifier": MarginProbaClassifier}
OVR_COMPONENTS = {**COMPONENTS, "ScoreRegressor": ScoreRegressor, "DecisionRegressor": DecisionRegressor,
              "Float32Regressor": Float32Regressor}


def cases():
    rng = np.random.default_rng(20261001)
    out = []
    for n_classes in (1, 2, 3, 5):
        for component in OVR_COMPONENTS:
            for label in ("int64", "float64"):
                n, p = 24, 3
                X = rng.normal(size=(n, p))
                codes = np.arange(n) % n_classes
                rng.shuffle(codes)
                X += codes[:, None] * 0.7
                pool = np.array([-3, 0, 2, 7, 11]) if label == "int64" else np.array([-1.0, 0.0, 2.0, 1e15, 4.0])
                y = pool[np.sort(rng.choice(5, n_classes, replace=False))][codes]
                Xt = rng.normal(size=(9, p)) * 1.5
                out.append(dict(name=f"{component}-k{n_classes}-{label}", component=component,
                                shift=float(rng.normal()) if n_classes == 2 else 0.0,
                                label=label, X=X, y=y, Xt=Xt))
    # Exact ties: duplicated training rows with symmetric classes, integer grid tests.
    X = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 1, 2, 3, 0, 3])
    Xt = np.array([[1, 1], [0, 1], [1, 0], [2, 1], [1, 2], [0, 0]], dtype=float)
    for component in OVR_COMPONENTS:
        out.append(dict(name=f"{component}-ties", component=component, shift=0.0, label="int64", X=X, y=y, Xt=Xt))
    # NaN features at inference propagate through np.maximum.
    Xn = np.array([[np.nan, 0.0], [0.0, np.nan], [0.5, 0.5]])
    out.append(dict(name="MarginClassifier-nan", component="MarginClassifier", shift=0.0, label="int64",
                    X=X, y=y, Xt=Xn))
    # Identical class centroids give an exactly controlled binary score.
    for component in ("ScoreRegressor", "DecisionRegressor", "Float32Regressor"):
        for shift in (0.0, 0.25, 0.5, 0.75):
            out.append(dict(name=f"{component}-threshold-{shift}", component=component,
                            shift=shift, label="int64", X=np.zeros((4, 1)),
                            y=np.array([-3, 7, -3, 7]), Xt=np.zeros((3, 1))))
    for component in OVR_COMPONENTS:
        for label in ("int64", "float32", "float64"):
            for n_labels in (2, 3, 5):
                X = rng.normal(size=(24, 3))
                y = rng.integers(0, 2, size=(24, n_labels))
                if n_labels == 3:
                    y[:, 0], y[:, 1] = 0, 1
                elif n_labels == 5:
                    y[:, -2], y[:, -1] = 0, 1
                out.append(dict(name=f"{component}-multilabel-{n_labels}-{label}",
                                component=component, shift=0.25, label=label,
                                X=X, y=y, Xt=rng.normal(size=(9, 3))))
        for value in (0, 1):
            out.append(dict(name=f"{component}-constant-labels-{value}",
                            component=component, shift=0.0, label="int64",
                            X=np.zeros((4, 1)), y=np.full((4, 2), value), Xt=np.zeros((3, 1))))
        for constant_first in (False, True):
            for shift in (0.0, 0.25, 0.5, 0.75):
                y = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
                if constant_first:
                    y[:, 0] = 0
                out.append(dict(name=f"{component}-multilabel-threshold-{constant_first}-{shift}",
                                component=component, shift=shift, label="int64",
                                X=np.zeros((4, 1)), y=y, Xt=np.zeros((3, 1))))
    return out


def reference(case):
    est = OVR_COMPONENTS[case["component"]](shift=case["shift"])
    model = multiclass.OneVsRestClassifier(est).fit(case["X"], case["y"])
    result = dict(classes=model.classes_.tolist(), n_estimators=len(model.estimators_), multilabel=int(model.multilabel_),
                  predict=model.predict(case["Xt"]).tolist(),
                  score=model.score(case["X"], case["y"]),
                  wscore=model.score(case["X"], case["y"], sample_weight=np.arange(1, len(case["y"]) + 1.0)))
    if all(hasattr(e, "decision_function") for e in model.estimators_):
        result["decision"] = np.asarray(model.decision_function(case["Xt"]), dtype=float).tolist()
    if all(hasattr(e, "predict_proba") for e in model.estimators_):
        result["proba"] = model.predict_proba(case["Xt"]).tolist()
    return result


def lit(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan[DType.float64]()"
    return repr(float(v))


def matrix(rows):
    return "[" + ", ".join("[" + ", ".join(lit(v) for v in r) + "]" for r in rows) + "]"


def driver(all_cases):
    lines = ['''from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import nan
from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import MarginClassifier, ProbaClassifier, MarginProbaClassifier, ScoreRegressor, DecisionRegressor, Float32Regressor


def mat(rows: List[List[Float64]]) raises -> NDArray[DType.float64]:
    var r = nm.zeros[DType.float64](Shape(len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            r[Item(i, j)] = rows[i][j]
    return r^


def vec[dtype: DType](values: List[Float64]) raises -> NDArray[dtype]:
    var r = nm.zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        r[Item(i)] = Scalar[dtype](values[i])
    return r^


def emit[dtype: DType](name: String, a: NDArray[dtype]) raises:
    var s = String('"') + name + '": ['
    if a.ndim == 1:
        for i in range(a.size):
            s += (", " if i else "") + String(Float64(a[Item(i)]))
    else:
        for i in range(a.shape[0]):
            s += (", " if i else "") + "["
            for j in range(a.shape[1]):
                s += (", " if j else "") + String(Float64(a[Item(i, j)]))
            s += "]"
    print(s + "],")

''']
    body = []
    for idx, c in enumerate(all_cases):
        lbl = "DType." + c["label"]
        target = (f"mat({matrix(c['y'])}).astype[{lbl}]()" if c["y"].ndim == 2
                  else f"vec[{lbl}]([{', '.join(lit(v) for v in c['y'])}])")
        body.append(f"# {c['name']}")
        body.append("@no_inline")
        body.append(f"def case_{idx}() raises:")
        body += [
            f"        var X = mat({matrix(c['X'])})",
            f"        var Xt = mat({matrix(c['Xt'])})",
            f"        var y = {target}",
            f"        var w = vec[DType.float64]([{', '.join(lit(v) for v in range(1, len(c['y']) + 1))}])",
            f"        var m = OneVsRestClassifier[{c['component']}, label={lbl}]({c['component']}(shift={lit(c['shift'])}))",
            "        m.fit(X, y)",
            f"        print('{{\"case\": {idx},')",
            "        emit('classes', m.classes_)",
            "        print('\"multilabel\":', Int(m.multilabel_()), ',')",
            "        print('\"n_estimators\":', len(m.estimators_), ',')",
            "        emit('predict', m.predict(Xt))",
            "        print('\"score\":', m.score(X, y), ',')",
            "        print('\"wscore\":', m.score(X, y, w.copy()), ',')",
        ]
        if hasattr(OVR_COMPONENTS[c["component"]], "decision_function"):
            body.append("        emit('decision', m.decision_function(Xt))")
        if hasattr(OVR_COMPONENTS[c["component"]], "predict_proba"):
            body.append("        emit('proba', m.predict_proba(Xt))")
        body.append("        print('\"end\": 0}')")
    # Keep cases out of one enormous function as the parity corpus grows.
    body = [line[4:] if line.startswith("        ") else line for line in body]
    body.append("def main() raises:")
    body.extend(f"    case_{idx}()" for idx in range(len(all_cases)))
    return lines[0] + "\n\n" + "\n".join(body) + "\n"


def parse(stdout):
    """Each case prints a JSON object across lines, closed by an "end" line."""
    out, cur = [], ""
    for line in stdout.strip().split("\n"):
        cur += line.replace(" nan", " NaN").replace("[nan", "[NaN").replace(" inf", " Infinity").replace("-inf", "-Infinity")
        if line.startswith('"end"'):
            out.append(json.loads(cur))
            cur = ""
    return out


def close(key, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rtol, atol = TOLERANCE[key]
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def max_error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if a.shape != b.shape or not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite]) / np.maximum(np.abs(a[finite]), 1e-300)))


def main():
    all_cases = cases()
    expected = [reference(c) for c in all_cases]
    source = OUT / "ovr_driver.mojo"
    source.write_text(driver(all_cases))
    binary = OUT / "ovr_driver"
    subprocess.run(["mojo", "build", "-I", ROOT / "builds/deps", "-I", ROOT / "tests/multiclass", source, "-o", binary], check=True)
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)
    assert len(got) == len(all_cases), (len(got), len(all_cases))
    failures = []
    worst = {"decision": 0.0, "proba": 0.0}
    for c, ref, out in zip(all_cases, expected, got):
        checks = {
            "classes": np.array_equal(np.asarray(ref["classes"], float), np.asarray(out["classes"], float)),
            "n_estimators": ref["n_estimators"] == out["n_estimators"],
            "multilabel": ref["multilabel"] == out["multilabel"],
            "predict": np.array_equal(np.asarray(ref["predict"], float), np.asarray(out["predict"], float)),
            "score": ref["score"] == out["score"],
            "wscore": math.isclose(ref["wscore"], out["wscore"], rel_tol=1e-15),
        }
        for key in ("decision", "proba"):
            if key in ref and key in out:
                checks[key] = close(key, ref[key], out[key])
                worst[key] = max(worst[key], max_error(ref[key], out[key]))
            elif (key in ref) != (key in out):
                # Constant-only fits can provide capabilities in sklearn that
                # are absent from the native component's compile-time type.
                all_constant = (len(ref["classes"]) == 1 if c["y"].ndim == 1
                                else all(np.unique(col).size == 1 for col in c["y"].T))
                checks[key + "-availability"] = key in ref and all_constant
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            failures.append((c["name"], bad))
    report = dict(sklearn=sklearn.__version__, revision=REVISION, cases=len(all_cases),
                  tolerance=TOLERANCE, max_relative_error=worst, failures=failures)
    (OUT / "ovr-parity-report.json").write_text(json.dumps(report, indent=2))
    if failures:
        for name, bad in failures:
            print("FAIL", name, bad)
        sys.exit(1)
    print(f"PASS: {len(all_cases)} OneVsRestClassifier cases match scikit-learn {sklearn.__version__};"
          f" max relative error {worst}")


if __name__ == "__main__":
    main()
