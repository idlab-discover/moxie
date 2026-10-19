"""Compare moxie OneVsOneClassifier with the pinned local scikit-learn.

Setup is shared with tests/ovr_parity.py (builds/reference-env). Run after
`pixi run build`:
    builds/reference-env/bin/python tests/multiclass/ovo_parity.py

scikit-learn's own OneVsOneClassifier wraps the Python twins of the Mojo test
components defined in ovr_parity.py. Python only creates fixtures, launches the
native driver and compares results; the library runtime is pure Mojo.
"""
from pathlib import Path
import json
import math
import subprocess
import sys

import numpy as np
import sklearn
from sklearn import multiclass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovr_parity import OVR_COMPONENTS as COMPONENTS, OUT, REVISION, ROOT, lit, matrix, parse  # noqa: E402

# Margin-component decision values repeat sklearn's Float64 operation order
# (votes + s / (3 * (|s| + 1)) with s accumulated pair by pair), so only FMA
# contraction in the native build can separate them. Probability components
# feed P(positive) through the fixture's exp, which differs from libm by up to
# ~1.2e-11 relative (see ovr_parity.py); the transform contracts that error by
# 1 / (3 (|s| + 1)^2) <= 1/3 per pair and sums at most n_classes - 1 pairs per
# column, so 5e-11 absolute bounds it for the 5-class cases here.
TOLERANCE = {"decision": (1e-12, 1e-14), "proba_decision": (0.0, 5e-11)}


def cases():
    rng = np.random.default_rng(20261002)
    out = []
    for n_classes in (2, 3, 4, 5):
        for component in COMPONENTS:
            for label in ("int64", "float64"):
                n, p = 30, 3
                X = rng.normal(size=(n, p))
                codes = np.arange(n) % n_classes
                rng.shuffle(codes)
                X += codes[:, None] * 0.6
                pool = np.array([-3, 0, 2, 7, 11]) if label == "int64" else np.array([-1.0, 0.0, 2.0, 1e15, 4.0])
                y = pool[np.sort(rng.choice(5, n_classes, replace=False))][codes]
                Xt = rng.normal(size=(11, p)) * 1.5 + 1.0
                out.append(dict(name=f"{component}-k{n_classes}-{label}", component=component,
                                shift=float(rng.normal()), label=label, X=X, y=y, Xt=Xt))
    # Unbalanced classes: pair subsets differ in size.
    X = rng.normal(size=(20, 2))
    y = np.array([0] * 9 + [1] * 2 + [2] * 6 + [3] * 3)
    X += y[:, None] * 0.8
    for component in COMPONENTS:
        out.append(dict(name=f"{component}-unbalanced", component=component, shift=0.0, label="int64",
                        X=X, y=y, Xt=rng.normal(size=(9, 2)) + 1.0))
    # Exact vote and confidence ties on an integer grid.
    X = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 1, 2, 3, 0, 3])
    Xt = np.array([[1, 1], [0, 1], [1, 0], [2, 1], [1, 2], [0, 0]], dtype=float)
    for component in COMPONENTS:
        out.append(dict(name=f"{component}-ties", component=component, shift=0.0, label="int64", X=X, y=y, Xt=Xt))
    # NaN features pass validation (ensure_all_finite=False) and reach np.argmax.
    Xn = np.array([[np.nan, 0.0], [0.0, np.nan], [0.5, 0.5]])
    for component in ("MarginClassifier", "MarginProbaClassifier"):
        out.append(dict(name=f"{component}-nan", component=component, shift=0.0, label="int64", X=X, y=y, Xt=Xn))
    # Exactly representable scores include 0/1 votes, fractions on both sides
    # of 0.5, and out-of-range values that must never be rounded into votes.
    X = np.array([[0.0], [2.0]])
    Xt = np.array([[0.75], [1.0], [1.0625], [1.125], [1.1875], [1.25], [1.3125], [1.5]])
    for component in ("ScoreRegressor", "DecisionRegressor", "Float32Regressor"):
        out.append(dict(name=f"{component}-exact-votes", component=component,
                        shift=0.0, label="int64", X=X, y=np.array([-3, 7]), Xt=Xt))
    return out


def reference(case):
    est = COMPONENTS[case["component"]](shift=case["shift"])
    model = multiclass.OneVsOneClassifier(est).fit(case["X"], case["y"])
    return dict(classes=model.classes_.tolist(), n_estimators=len(model.estimators_),
                predict=model.predict(case["Xt"]).tolist(),
                decision=np.asarray(model.decision_function(case["Xt"]), dtype=float).tolist(),
                score=model.score(case["X"], case["y"]),
                wscore=model.score(case["X"], case["y"], sample_weight=np.arange(1, len(case["y"]) + 1.0)),
                pairwise=model.pairwise_indices_ is None)


def driver(all_cases):
    header = '''from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import nan
from moxie.multiclass import OneVsOneClassifier
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
'''
    body = []
    for idx, c in enumerate(all_cases):
        lbl = "DType.int64" if c["label"] == "int64" else "DType.float64"
        body += [
            f"# {c['name']}",
            "@no_inline",
            f"def case_{idx}() raises:",
            f"        var X = mat({matrix(c['X'])})",
            f"        var Xt = mat({matrix(c['Xt'])})",
            f"        var y = vec[{lbl}]([{', '.join(lit(v) for v in c['y'])}])",
            f"        var w = vec[DType.float64]([{', '.join(lit(v) for v in range(1, len(c['y']) + 1))}])",
            f"        var m = OneVsOneClassifier[{c['component']}, label={lbl}]({c['component']}(shift={lit(c['shift'])}))",
            "        m.fit(X, y)",
            f"        print('{{\"case\": {idx},')",
            "        emit('classes', m.classes_)",
            "        print('\"n_estimators\":', len(m.estimators_), ',')",
            "        print('\"pairwise\":', 'true' if not m.pairwise_indices_ else 'false', ',')",
            "        emit('predict', m.predict(Xt))",
            "        emit('decision', m.decision_function(Xt))",
            "        print('\"score\":', m.score(X, y), ',')",
            "        print('\"wscore\":', m.score(X, y, w.copy()), ',')",
            "        print('\"end\": 0}')",
        ]
    body = [line[4:] if line.startswith("        ") else line for line in body]
    body += ["def main() raises:"] + [f"    case_{idx}()" for idx in range(len(all_cases))]
    return header + "\n\n" + "\n".join(body) + "\n"


def close(key, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rtol, atol = TOLERANCE[key]
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def main():
    all_cases = cases()
    expected = [reference(c) for c in all_cases]
    source = OUT / "ovo_driver.mojo"
    source.write_text(driver(all_cases))
    binary = OUT / "ovo_driver"
    subprocess.run(["mojo", "build", "-I", ROOT / "builds/deps", "-I", ROOT / "tests/multiclass", source, "-o", binary], check=True)
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)
    assert len(got) == len(all_cases), (len(got), len(all_cases))
    failures = []
    worst = {"decision": 0.0, "proba_decision": 0.0}
    exact_decisions = 0
    for c, ref, out in zip(all_cases, expected, got):
        key = "proba_decision" if c["component"] == "ProbaClassifier" else "decision"
        checks = {
            "classes": np.array_equal(np.asarray(ref["classes"], float), np.asarray(out["classes"], float)),
            "n_estimators": ref["n_estimators"] == out["n_estimators"],
            "pairwise": ref["pairwise"] == out["pairwise"],
            "predict": np.array_equal(np.asarray(ref["predict"], float), np.asarray(out["predict"], float)),
            "decision": close(key, ref["decision"], out["decision"]),
            "score": ref["score"] == out["score"],
            "wscore": math.isclose(ref["wscore"], out["wscore"], rel_tol=1e-15),
        }
        a, b = np.asarray(ref["decision"], float), np.asarray(out["decision"], float)
        exact_decisions += a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        worst[key] = max(worst[key], float(np.nanmax(np.abs(a - b))) if a.shape == b.shape and np.isfinite(a - b).any() else 0.0)
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            failures.append((c["name"], bad))
    report = dict(sklearn=sklearn.__version__, revision=REVISION, cases=len(all_cases),
                  tolerance=TOLERANCE, max_absolute_error=worst,
                  bitwise_equal_decisions=exact_decisions, failures=failures)
    (OUT / "ovo-parity-report.json").write_text(json.dumps(report, indent=2))
    if failures:
        for name, bad in failures:
            print("FAIL", name, bad)
        sys.exit(1)
    print(f"PASS: {len(all_cases)} OneVsOneClassifier cases match scikit-learn {sklearn.__version__};"
          f" {exact_decisions} bitwise-equal decision arrays; max absolute error {worst}")


if __name__ == "__main__":
    main()
