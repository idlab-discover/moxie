"""Compare the multiclass wrappers around production Moxie components to sklearn.

The other runners use paired test doubles to isolate wrapper logic. This one
wraps real Moxie estimators and their sklearn counterparts:

- `moxie.linear_model.RidgeClassifier` (decision_function component),
- `moxie.tree.DecisionTreeClassifier` through the dense
  `moxie.feature_selection.DecisionTreeClassifierImportance` adapter
  (predict_proba component; the tree's own API returns List probabilities), and
- `moxie.svm.LinearSVC` and
  `moxie.discriminant_analysis.LinearDiscriminantAnalysis` through
  `BinaryAdapter` for the native List targets.

Cases cover OvR, OvO and OutputCode; 2, 3 and 5 classes; int64, float64 and
string labels; Float32 and Float64 features; seeded OutputCode codebooks.

Run after `pixi run python tools/multiclass_checks.py build`:
    builds/reference-env/bin/python tests/multiclass/production_parity.py

Tolerances:
- classes, component counts, codebooks, predictions and scores must match exactly;
- Float64 Ridge decisions: 1e-9 relative / 1e-11 absolute. Moxie solves the
  ridge system with its own Cholesky kernels while sklearn calls LAPACK
  (`scipy.linalg.solve`); the solves differ by reduction order only;
- Float32 Ridge decisions: 2e-4 relative / 1e-5 absolute, float32 epsilon
  (1.2e-7) amplified by the conditioning of the centered normal equations;
- tree probabilities are leaf class frequencies (exact ratios of small
  integers) and must agree within 1e-15; OvO tree decisions combine them with
  votes through one division, allowed 1e-14 relative;
- LinearSVC uses the primal solver with tol=1e-9. Allow 1e-7 relative / 1e-8
  absolute for decision values because of reduction order and stopping;
- LDA uses the default SVD solver. Allow 1e-9 relative / 1e-11 absolute for
  Float64 responses across SVD implementations, and 1e-5 relative / 5e-6
  absolute for Float32 responses because Moxie accumulates in Float64.
"""
import argparse
import inspect
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
from sklearn import discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier, _predict_binary
from sklearn.svm import LinearSVC, _classes as svm_classes
from sklearn.tree import DecisionTreeClassifier

from ovr_parity import OUT, REFERENCE, REVISION, ROOT, driver, lit, matrix, parse

TOLERANCE = {
    "ridge64": (1e-9, 1e-11),
    "ridge32": (2e-4, 1e-5),
    "tree": (1e-14, 1e-15),
    "linear-svc": (1e-7, 1e-8),
    "lda64": (1e-9, 1e-11),
    "lda32": (1e-5, 5e-6),
}

COMPONENTS = {
    "ridge": (lambda: RidgeClassifier(),
              "RidgeClassifier[{dtype}]()"),
    "ridge-noint": (lambda: RidgeClassifier(alpha=0.1, fit_intercept=False),
                    "RidgeClassifier[{dtype}](0.1, fit_intercept=False)"),
    "tree": (lambda: DecisionTreeClassifier(random_state=0),
             "DecisionTreeClassifierImportance[{dtype}](DecisionTreeClassifier[DType.int64](random_state=0))"),
    "tree-depth2": (lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
                    "DecisionTreeClassifierImportance[{dtype}](DecisionTreeClassifier[DType.int64](max_depth=2, random_state=0))"),
    "linear-svc": (lambda: LinearSVC(dual=False, tol=1e-9, random_state=0),
                   "BinaryAdapter(LinearSVC(dual=False, tol=1e-9, random_state=0))"),
    "lda": (LinearDiscriminantAnalysis,
            "BinaryAdapter(LinearDiscriminantAnalysis[{dtype}]())"),
}
COMPONENT_TYPE = {
    "ridge": "RidgeClassifier[{dtype}]", "ridge-noint": "RidgeClassifier[{dtype}]",
    "tree": "DecisionTreeClassifierImportance[{dtype}]",
    "tree-depth2": "DecisionTreeClassifierImportance[{dtype}]",
    "linear-svc": "BinaryAdapter[LinearSVC[]]",
    "lda": "BinaryAdapter[LinearDiscriminantAnalysis[{dtype}]]",
}
STRINGS = ["setosa", "versicolor", "virginica", "", "zeta"]


def cases(components):
    rng = np.random.default_rng(20261021)
    result = []
    for wrapper in ("ovr", "ovo", "oc"):
        for component in components:
            for k in (2, 3, 5):
                for labels in ("int64", "float64", "string", "float32-features"):
                    if labels == "float32-features" and component == "linear-svc":
                        continue
                    if labels != "int64" and k != 3:
                        continue
                    n = 12 * k
                    y = np.tile(np.arange(k), 12)
                    rng.shuffle(y)
                    X = rng.normal(size=(n, 4)) + y[:, None] * np.array([0.8, -0.5, 0.3, 0.0])
                    Xt = rng.normal(size=(15, 4)) + rng.integers(0, k, 15)[:, None] * 0.6
                    dtype = "float32" if labels == "float32-features" else "float64"
                    if dtype == "float32":
                        X, Xt = X.astype(np.float32), Xt.astype(np.float32)
                    if labels == "string":
                        target = np.array([STRINGS[v] for v in y])
                    elif labels == "float64":
                        target = (y * 3 - 7).astype(np.float64)
                    else:
                        target = y * 3 - 7
                    for code_size, seed in (((1.5, 0), (2.0, 3)) if wrapper == "oc" else ((None, None),)):
                        result.append(dict(
                            name=f"{wrapper}-{component}-k{k}-{labels}" + (f"-size{code_size}-seed{seed}" if seed is not None else ""),
                            wrapper=wrapper, component=component, labels=labels, dtype=dtype,
                            code_size=code_size, seed=seed, X=X, Xt=Xt, y=target))
    return result


def reference(c):
    base = COMPONENTS[c["component"]][0]()
    if c["wrapper"] == "ovr":
        model = OneVsRestClassifier(base)
    elif c["wrapper"] == "ovo":
        model = OneVsOneClassifier(base)
    else:
        model = OutputCodeClassifier(base, code_size=c["code_size"], random_state=c["seed"])
    model.fit(c["X"], c["y"])
    out = dict(classes=[str(v) if c["labels"] == "string" else float(v) for v in model.classes_],
               n_estimators=len(model.estimators_),
               predict=[str(v) if c["labels"] == "string" else float(v) for v in model.predict(c["Xt"])],
               score=float(model.score(c["X"], c["y"])))
    if c["wrapper"] in ("ovr", "ovo") and hasattr(model, "decision_function"):
        out["decision"] = np.asarray(model.decision_function(c["Xt"]), dtype=float).tolist()
    if c["wrapper"] == "ovr" and hasattr(model, "predict_proba"):
        out["proba"] = model.predict_proba(c["Xt"]).tolist()
    if c["wrapper"] == "oc":
        out["code_book"] = model.code_book_.tolist()
        out["scores"] = np.array([_predict_binary(e, c["Xt"]) for e in model.estimators_]).T.tolist()
    return out


def exact_tie(ref, row, a, b):
    """Whether codewords a and b are both nearest to row in exact arithmetic."""
    y = [Fraction(v) for v in ref["scores"][row]]
    dist = [sum((v - Fraction(bit)) ** 2 for v, bit in zip(y, word)) for word in ref["code_book"]]
    index = {label: i for i, label in enumerate(ref["classes"])}
    return dist[index[a]] == dist[index[b]] == min(dist)


def native_source(all_cases, expected):
    header = driver([]).split("def main() raises:")[0]
    header += '''
from moxie.multiclass import BinaryAdapter, OneVsOneClassifier, OutputCodeClassifier, StringLabels, StringTarget
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.linear_model import RidgeClassifier
from moxie.svm import LinearSVC
from moxie.tree import DecisionTreeClassifier
from moxie.feature_selection import DecisionTreeClassifierImportance


def emit(name: String, values: List[String]):
    var line = String('"') + name + '": ['
    for i in range(len(values)):
        line += (", " if i else "") + String('"') + values[i] + '"'
    print(line + "],")


def emit(name: String, y: StringTarget) raises:
    emit(name, y.values())

'''
    body = []
    for i, (c, ref) in enumerate(zip(all_cases, expected)):
        dtype = "DType." + c["dtype"]
        ctype = COMPONENT_TYPE[c["component"]].format(dtype=dtype)
        component = COMPONENTS[c["component"]][1].format(dtype=dtype)
        cls = {"ovr": "OneVsRestClassifier", "ovo": "OneVsOneClassifier", "oc": "OutputCodeClassifier"}[c["wrapper"]]
        if c["labels"] == "string":
            params = f"{ctype}, {dtype}, kind=StringLabels"
            target = "StringTarget([" + ", ".join(json.dumps(str(v)) for v in c["y"]) + "])"
        else:
            label = "DType.float64" if c["labels"] == "float64" else "DType.int64"
            params = f"{ctype}, {dtype}, {label}"
            target = f"vec[{label}]([{', '.join(lit(v) for v in c['y'])}])"
        extra = (f", code_size={lit(c['code_size'])}, random_state={c['seed']}" if c["wrapper"] == "oc" else "")
        body += [
            f"# {c['name']}", "@no_inline", f"def case_{i}() raises:",
            f"    var X = mat({matrix(c['X'])}).astype[{dtype}]()",
            f"    var Xt = mat({matrix(c['Xt'])}).astype[{dtype}]()",
            f"    var y = {target}",
            f"    var m = {cls}[{params}]({component}{extra})",
            "    _ = m.fit(X, y)",
            f"    print('{{\"case\": {i},')",
            "    emit('classes', m.classes_)",
            "    print('\"n_estimators\":', len(m.estimators_), ',')",
            "    emit('predict', m.predict(Xt))",
            "    print('\"score\":', m.score(X, y), ',')",
        ]
        if "decision" in ref:
            body.append("    emit('decision', m.decision_function(Xt))")
        if "proba" in ref:
            body.append("    emit('proba', m.predict_proba(Xt))")
        if "code_book" in ref:
            body.append("    emit('code_book', m.code_book_)")
        body.append("    print('\"end\": 0}')")
    body.append("def main() raises:")
    body += [f"    case_{i}()" for i in range(len(all_cases))]
    return header + "\n".join(body) + "\n"


def tolerance(c, key):
    if c["component"] == "linear-svc":
        return TOLERANCE["linear-svc"]
    if c["component"] == "lda":
        return TOLERANCE["lda32" if c["dtype"] == "float32" else "lda64"]
    if c["component"].startswith("tree"):
        return (1e-14, 1e-15) if key == "decision" else TOLERANCE["tree"]
    return TOLERANCE["ridge32" if c["dtype"] == "float32" else "ridge64"]


def run_native(group, all_cases, expected):
    source = OUT / f"production_{group}_driver.mojo"
    source.write_text(native_source(all_cases, expected))
    binary = OUT / f"production_{group}_driver"
    subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", ROOT / "builds/deps",
                    "-I", ROOT / "tests/multiclass", source, "-o", binary], check=True)
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)
    assert len(got) == len(all_cases), (len(got), len(all_cases))
    return got


def main():
    groups = {
        "ridge-tree": ("ridge", "ridge-noint", "tree", "tree-depth2"),
        "linear-svc": ("linear-svc",),
        "lda": ("lda",),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("all", *groups), default="all")
    selection = parser.parse_args().group
    all_cases, expected, got = [], [], []
    for group, components in groups.items():
        if selection not in ("all", group):
            continue
        if group == "linear-svc":
            assert Path(inspect.getfile(svm_classes)).read_bytes() == (
                REFERENCE / "sklearn/svm/_classes.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        if group == "lda":
            assert Path(inspect.getfile(discriminant_analysis)).read_bytes() == (
                REFERENCE / "sklearn/discriminant_analysis.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        batch = cases(components)
        ref = [reference(c) for c in batch]
        got.extend(run_native(group, batch, ref))
        all_cases.extend(batch)
        expected.extend(ref)
    failures, worst, tied_rows = [], {}, []
    for c, ref, out in zip(all_cases, expected, got):
        checks = {key: ref[key] == out[key] for key in ("classes", "n_estimators", "predict", "score")}
        if "code_book" in ref and not checks["predict"] and len(ref["predict"]) == len(out["predict"]):
            # sklearn resolves exact codeword ties by BLAS rounding (see output_code.md).
            rows = [i for i, (a, b) in enumerate(zip(ref["predict"], out["predict"])) if a != b]
            if all(exact_tie(ref, i, ref["predict"][i], out["predict"][i]) for i in rows):
                checks["predict"] = True
                tied_rows.append((c["name"], rows))
        if "code_book" in ref:
            checks["code_book"] = np.array_equal(ref["code_book"], out["code_book"])
        for key in ("decision", "proba"):
            if key in ref:
                a, b = np.asarray(ref[key], float), np.asarray(out.get(key, []), float)
                rtol, atol = tolerance(c, key)
                checks[key] = a.shape == b.shape and np.allclose(b, a, rtol=rtol, atol=atol)
                if a.shape == b.shape:
                    component = c["component"].removesuffix("-noint").removesuffix("-depth2")
                    group = f"{key}-{component}-{c['dtype']}"
                    worst[group] = max(worst.get(group, 0.0), float(np.max(np.abs(a - b))))
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            failures.append((c["name"], bad))
    report = dict(sklearn=sklearn.__version__, revision=REVISION, cases=len(all_cases),
                  tolerance=TOLERANCE, max_absolute_error=worst, exact_ties=tied_rows,
                  failures=failures)
    report_name = "production-parity-report.json" if selection == "all" else f"production-{selection}-report.json"
    (OUT / report_name).write_text(json.dumps(report, indent=2))
    if failures:
        for name, bad in failures:
            print("FAIL", name, bad)
        sys.exit(1)
    print(f"PASS: {len(all_cases)} production-component cases ({selection}) "
          f"match scikit-learn {sklearn.__version__}; max absolute error {worst}; "
          f"OutputCode rows resolved differently at exact codeword ties: {tied_rows}")


if __name__ == "__main__":
    main()
