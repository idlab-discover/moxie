"""Compare the multiclass wrappers around production Moxie components to sklearn.

The other runners use paired test doubles to isolate wrapper logic. This one
wraps real Moxie estimators and their sklearn counterparts:

- `moxie.linear_model.RidgeClassifier` (decision_function component),
- `moxie.tree.DecisionTreeClassifier` with native Float32 features, Int64
  component targets and one Float64 probability matrix per output, and
- `moxie.svm.SVC` and `moxie.svm.LinearSVC`,
  `moxie.discriminant_analysis.LinearDiscriminantAnalysis` and
  `QuadraticDiscriminantAnalysis`, and
  `moxie.naive_bayes.GaussianNB` through `BinaryAdapter` for the native List
  targets. GaussianNB also exercises incremental OvR and OvO updates.
- `moxie.dummy.DummyClassifier` with integer-valued native labels through
  `BinaryAdapter`.
- `moxie.neighbors.KNeighborsClassifier` with Int64 array component targets
  and one native Float64 probability matrix per output.

Cases cover OvR, OvO and OutputCode; 2, 3 and 5 classes; int64, float64 and
string labels; Float32 and Float64 features; seeded OutputCode codebooks.

Run with the locked toolchain and current sources assembled in builds/deps:
    builds/reference-env/bin/python tests/multiclass/production_parity.py --group svc

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
- Dummy prior probabilities use the same tolerance for small count ratios.
- LinearSVC uses the primal solver with tol=1e-9. Allow 1e-7 relative / 1e-8
  absolute for decision values because of reduction order and stopping;
- SVC uses the SVM family's public parity tolerance (3e-5 relative / 3e-6
  absolute) for LIBSVM solver, kernel and probability calibration rounding.
- LDA uses the default SVD solver. Allow 1e-9 relative / 1e-11 absolute for
  Float64 responses across SVD implementations, and 1e-5 relative / 5e-6
  absolute for Float32 responses because Moxie accumulates in Float64.
- QDA uses the default SVD solver and the eigen solver with fixed shrinkage.
  Allow the same tolerances as LDA for factorization differences and
  Float32 rounding of fitted parameters and responses.
- GaussianNB allows 1e-9 relative / 1e-11 absolute for Float64 responses,
  and 1e-5 relative / 5e-6 absolute for Float32 responses because the
  native estimator stores its running moments and computes likelihoods in
  Float32, while sklearn retains Float64 learned means and variances.
- Uniform KNN votes use the tree probability tolerance. Distance weighting
  allows 1e-6 relative / 1e-7 absolute for Float32, where Moxie retains
  distances in the feature dtype before computing Float64 weights. Float64
  distance weighting allows 1e-12 relative / 1e-14 absolute for rounding.
"""
import argparse
import inspect
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import sklearn
from sklearn import discriminant_analysis, naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier, _predict_binary
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, _classification as neighbors_classification
from sklearn.svm import SVC, LinearSVC, _classes as svm_classes
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _classes as tree_classes

from ovr_parity import OUT, REFERENCE, REVISION, ROOT, driver, lit, matrix, parse

TOLERANCE = {
    "ridge64": (1e-9, 1e-11),
    "ridge32": (2e-4, 1e-5),
    "tree": (1e-14, 1e-15),
    "linear-svc": (1e-7, 1e-8),
    "svc": (3e-5, 3e-6),
    "lda64": (1e-9, 1e-11),
    "lda32": (1e-5, 5e-6),
    "gaussian-nb64": (1e-9, 1e-11),
    "gaussian-nb32": (1e-5, 5e-6),
    "knn-distance64": (1e-12, 1e-14),
    "knn-distance32": (1e-6, 1e-7),
}

COMPONENTS = {
    "ridge": (lambda: RidgeClassifier(),
              "RidgeClassifier[{dtype}]()"),
    "ridge-noint": (lambda: RidgeClassifier(alpha=0.1, fit_intercept=False),
                    "RidgeClassifier[{dtype}](0.1, fit_intercept=False)"),
    "tree": (lambda: DecisionTreeClassifier(random_state=0),
             "DecisionTreeClassifier[DType.int64](random_state=0)"),
    "tree-depth2": (lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
                    "DecisionTreeClassifier[DType.int64](max_depth=2, random_state=0)"),
    "linear-svc": (lambda: LinearSVC(dual=False, tol=1e-9, random_state=0),
                   "BinaryAdapter(LinearSVC(dual=False, tol=1e-9, random_state=0))"),
    "svc": (lambda: SVC(tol=1e-6, random_state=0),
            "BinaryAdapter(SVC(tol=1e-6, random_state=0))"),
    "svc-probability": (lambda: SVC(probability=True, tol=1e-6, random_state=0),
                        "BinaryAdapter(SVC(probability=True, tol=1e-6, random_state=0))"),
    "svc-precomputed": (lambda: SVC(kernel="precomputed", tol=1e-6, random_state=0),
                        'BinaryAdapter(SVC(kernel="precomputed", tol=1e-6, random_state=0))'),
    "lda": (LinearDiscriminantAnalysis,
            "BinaryAdapter(LinearDiscriminantAnalysis[{dtype}]())"),
    "qda": (QuadraticDiscriminantAnalysis,
            "BinaryAdapter(QuadraticDiscriminantAnalysis[{dtype}]())"),
    "qda-eigen": (lambda: QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.1),
                  'BinaryAdapter(QuadraticDiscriminantAnalysis[{dtype}](solver="eigen", shrinkage=0.1))'),
    "gaussian-nb": (GaussianNB,
                    "BinaryAdapter(GaussianNB[IntListDomain, {dtype}]())"),
    "dummy": (DummyClassifier,
              "BinaryAdapter(DummyClassifier[dtype={dtype}]())"),
    "knn": (lambda: KNeighborsClassifier(n_neighbors=3),
            "KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]](3)"),
    "knn-distance": (lambda: KNeighborsClassifier(n_neighbors=3, weights="distance"),
                     'KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]](3, weights="distance")'),
    "knn-precomputed": (
        lambda: KNeighborsClassifier(n_neighbors=3, metric="precomputed"),
        'KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]](3, metric="precomputed")',
    ),
    "knn-precomputed-distance": (
        lambda: KNeighborsClassifier(n_neighbors=3, metric="precomputed", weights="distance"),
        'KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]](3, metric="precomputed", weights="distance")',
    ),
}
COMPONENT_TYPE = {
    "ridge": "RidgeClassifier[{dtype}]", "ridge-noint": "RidgeClassifier[{dtype}]",
    "tree": "DecisionTreeClassifier[DType.int64]",
    "tree-depth2": "DecisionTreeClassifier[DType.int64]",
    "linear-svc": "BinaryAdapter[LinearSVC[]]",
    "svc": "BinaryAdapter[SVC[]]",
    "svc-probability": "BinaryAdapter[SVC[]]",
    "svc-precomputed": "BinaryAdapter[SVC[]]",
    "lda": "BinaryAdapter[LinearDiscriminantAnalysis[{dtype}]]",
    "qda": "BinaryAdapter[QuadraticDiscriminantAnalysis[{dtype}]]",
    "qda-eigen": "BinaryAdapter[QuadraticDiscriminantAnalysis[{dtype}]]",
    "gaussian-nb": "BinaryAdapter[GaussianNB[IntListDomain, {dtype}]]",
    "dummy": "BinaryAdapter[DummyClassifier[dtype={dtype}]]",
    "knn": "KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]]",
    "knn-distance": "KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]]",
    "knn-precomputed": "KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]]",
    "knn-precomputed-distance": "KNeighborsClassifier[{dtype}, Y=NDArray[DType.int64]]",
}
SVC_COMPONENTS = ("svc", "svc-probability", "svc-precomputed")
KNN_PRECOMPUTED = ("knn-precomputed", "knn-precomputed-distance")
PAIRWISE_COMPONENTS = ("svc-precomputed", *KNN_PRECOMPUTED)
OUTPUT_CODE_RESPONSE = {
    "tree": ("predict_proba", "response[0][Item(i, 1)]"),
    "tree-depth2": ("predict_proba", "response[0][Item(i, 1)]"),
    "gaussian-nb": ("predict_proba", "response[Item(i, 1)]"),
    "dummy": ("predict_proba", "response[Item(i, 1)]"),
    "knn": ("predict_proba", "response[0][Item(i, 1)]"),
    "knn-distance": ("predict_proba", "response[0][Item(i, 1)]"),
    "svc": ("decision_function", "response[Item(i)]"),
    "qda": ("decision_function", "Float64(response[Item(i)])"),
    "qda-eigen": ("decision_function", "Float64(response[Item(i)])"),
}
STRINGS = ["setosa", "versicolor", "virginica", "", "zeta"]


def cases(components):
    rng = np.random.default_rng(20261021)
    result = []
    for wrapper in ("ovr", "ovo", "oc"):
        for component in components:
            if component == "svc-probability" and wrapper != "ovr":
                continue
            if component in PAIRWISE_COMPONENTS and wrapper != "ovo":
                continue
            for k in (2, 3, 5):
                for labels in ("int64", "float64", "string", "float32-features"):
                    if labels == "float32-features" and component in ("linear-svc", *SVC_COMPONENTS):
                        continue
                    if labels != "int64" and k != 3:
                        continue
                    repeats = 6 if component in KNN_PRECOMPUTED else 12
                    n = repeats * k
                    y = np.tile(np.arange(k), repeats)
                    rng.shuffle(y)
                    X = rng.normal(size=(n, 4)) + y[:, None] * np.array([0.8, -0.5, 0.3, 0.0])
                    Xt = rng.normal(size=(15, 4)) + rng.integers(0, k, 15)[:, None] * 0.6
                    dtype = "float32" if labels == "float32-features" or component.startswith("tree") else "float64"
                    if labels == "float32-features":
                        X, Xt = X.astype(np.float32), Xt.astype(np.float32)
                    if labels == "string":
                        target = np.array([STRINGS[v] for v in y])
                    elif labels == "float64":
                        target = (y * 3 - 7).astype(np.float64)
                    else:
                        target = y * 3 - 7
                    if component == "svc-precomputed":
                        X, Xt = X @ X.T, Xt @ X.T
                    if component in KNN_PRECOMPUTED:
                        X, Xt = (
                            np.linalg.norm(rows[:, None, :] - X[None, :, :], axis=2)
                            for rows in (X, Xt)
                        )
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
    target = c["y"]
    decoded_classes = None
    if c.get("incremental"):
        mid = len(target) // 2
        if c["wrapper"] == "ovo" and c["labels"] == "string":
            # Pinned sklearn uses zeros_like(y) for OvO batch codes, producing
            # string codes that GaussianNB rejects against numeric classes.
            with np.testing.assert_raises_regex(ValueError, "initial classes"):
                model.partial_fit(c["X"][:mid], target[:mid], classes=np.unique(target))
            decoded_classes, target = np.unique(target, return_inverse=True)
            model = OneVsOneClassifier(base)
        model.partial_fit(c["X"][:mid], target[:mid], classes=np.unique(target))
        model.partial_fit(c["X"][mid:], target[mid:])
    else:
        with catch_warnings():
            if c["component"] == "svc-probability":
                filterwarnings("ignore", r"The `probability` parameter was deprecated", FutureWarning)
            model.fit(c["X"], target)
    classes, predict = model.classes_, model.predict(c["Xt"])
    if decoded_classes is not None:
        classes, predict = decoded_classes[classes], decoded_classes[predict]
    convert = str if c["labels"] == "string" else float
    out = dict(classes=[convert(v) for v in classes],
               n_estimators=len(model.estimators_),
               predict=[convert(v) for v in predict],
               score=float(model.score(c["X"], target)))
    if c["wrapper"] in ("ovr", "ovo") and hasattr(model, "decision_function"):
        out["decision"] = np.asarray(model.decision_function(c["Xt"]), dtype=float).tolist()
    if c["wrapper"] == "ovr" and c["component"] == "svc":
        assert not hasattr(model, "predict_proba")
    if c["wrapper"] == "ovr" and hasattr(model, "predict_proba"):
        out["proba"] = model.predict_proba(c["Xt"]).tolist()
    if c["component"] in PAIRWISE_COMPONENTS:
        out["pairwise_indices"] = [indices.tolist() for indices in model.pairwise_indices_]
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


def target_source(values, label):
    if label is None:
        return "StringTarget([" + ", ".join(json.dumps(str(v)) for v in values) + "])"
    return f"vec[{label}]([{', '.join(lit(v) for v in values)}])"


def native_source(all_cases, expected):
    header = driver([]).split("def main() raises:")[0]
    header += '''
from moxie.multiclass import BinaryAdapter, OneVsOneClassifier, OutputCodeClassifier, StringLabels, StringTarget
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from moxie.dummy import DummyClassifier
from moxie.linear_model import RidgeClassifier
from moxie.naive_bayes import GaussianNB, IntListDomain
from moxie.neighbors import KNeighborsClassifier
from moxie.svm import LinearSVC, SVC
from moxie.tree import DecisionTreeClassifier
from std.testing import assert_raises


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
            label = None
        else:
            label = "DType.float64" if c["labels"] == "float64" else "DType.int64"
            params = f"{ctype}, {dtype}, {label}"

        extra = (f", code_size={lit(c['code_size'])}, random_state={c['seed']}" if c["wrapper"] == "oc" else "")
        body += [
            f"# {c['name']}", "@no_inline", f"def case_{i}() raises:",
            f"    var X = mat({matrix(c['X'])}).astype[{dtype}]()",
            f"    var Xt = mat({matrix(c['Xt'])}).astype[{dtype}]()",
            f"    var y = {target_source(c['y'], label)}",
            f"    var m = {cls}[{params}]({component}{extra})",
        ]
        if c.get("incremental"):
            mid = len(c["y"]) // 2
            known = "StringTarget" if label is None else f"NDArray[{label}]"
            first = matrix(c["X"][:mid])
            second = matrix(c["X"][mid:])
            body += [
                f"    var classes = Optional[{known}]({target_source(np.unique(c['y']), label)})",
                f"    _ = m.partial_fit(mat({first}).astype[{dtype}](), {target_source(c['y'][:mid], label)}, classes^)",
                f"    _ = m.partial_fit(mat({second}).astype[{dtype}](), {target_source(c['y'][mid:], label)})",
            ]
        else:
            body.append("    _ = m.fit(X, y)")
        body += [
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
        elif c["wrapper"] == "ovr" and c["component"] == "svc":
            body += [
                '    with assert_raises(contains="AttributeError: predict_proba requires probability=True"):',
                "        _ = m.predict_proba(Xt)",
            ]
        if "pairwise_indices" in ref:
            body.append("""    print('"n_pairs":', len(m.pairwise_indices_.value()), ',')""")
            for pair in range(len(ref["pairwise_indices"])):
                body.append(f"    emit('pairwise_{pair}', m.pairwise_indices_.value()[{pair}])")
        if "code_book" in ref:
            body.append("    emit('code_book', m.code_book_)")
            if c["component"] in OUTPUT_CODE_RESPONSE:
                method, value = OUTPUT_CODE_RESPONSE[c["component"]]
                if c["component"] == "gaussian-nb" and c["dtype"] == "float32":
                    value = f"Float64({value})"
                body += [
                    "    var scores = NDArray[DType.float64](Shape(Xt.shape[0], len(m.estimators_)))",
                    "    for k in range(len(m.estimators_)):",
                    "        ref fitted = m.estimators_[k]",
                    "        if fitted.is_constant():",
                    "            for i in range(Xt.shape[0]):",
                    "                scores[Item(i, k)] = Float64(fitted.constant())",
                    "        else:",
                    f"            var response = fitted.estimator().{method}(Xt)",
                    "            for i in range(Xt.shape[0]):",
                    f"                scores[Item(i, k)] = {value}",
                    "    emit('scores', scores)",
                ]
        body.append("    print('\"end\": 0}')")
    body.append("def main() raises:")
    body += [f"    case_{i}()" for i in range(len(all_cases))]
    return header + "\n".join(body) + "\n"


def tolerance(c, key):
    if c["component"] == "linear-svc":
        return TOLERANCE["linear-svc"]
    if c["component"] in SVC_COMPONENTS:
        return TOLERANCE["svc"]
    if c["component"] in ("lda", "qda", "qda-eigen"):
        return TOLERANCE["lda32" if c["dtype"] == "float32" else "lda64"]
    if c["component"] == "gaussian-nb":
        return TOLERANCE["gaussian-nb32" if c["dtype"] == "float32" else "gaussian-nb64"]
    if c["component"] in ("dummy", "knn", "knn-precomputed"):
        return TOLERANCE["tree"]
    if c["component"] in ("knn-distance", "knn-precomputed-distance"):
        return TOLERANCE["knn-distance32" if c["dtype"] == "float32" else "knn-distance64"]
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
        "tree": ("tree", "tree-depth2"),
        "linear-svc": ("linear-svc",),
        "svc": SVC_COMPONENTS,
        "lda": ("lda",),
        "qda": ("qda", "qda-eigen"),
        "gaussian-nb": ("gaussian-nb",),
        "dummy": ("dummy",),
        "knn": ("knn", "knn-distance"),
        "knn-precomputed": KNN_PRECOMPUTED,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("all", *groups), default="all")
    selection = parser.parse_args().group
    all_cases, expected, got = [], [], []
    for group, components in groups.items():
        if selection == "all" and group == "tree":
            continue  # The combined ridge-tree group already covers native trees.
        if selection not in ("all", group):
            continue
        if group in ("ridge-tree", "tree"):
            assert Path(inspect.getfile(tree_classes)).read_bytes() == (
                REFERENCE / "sklearn/tree/_classes.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        if group in ("linear-svc", "svc"):
            assert Path(inspect.getfile(svm_classes)).read_bytes() == (
                REFERENCE / "sklearn/svm/_classes.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        if group in ("lda", "qda"):
            assert Path(inspect.getfile(discriminant_analysis)).read_bytes() == (
                REFERENCE / "sklearn/discriminant_analysis.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        if group in ("knn", "knn-precomputed"):
            assert Path(inspect.getfile(neighbors_classification)).read_bytes() == (
                REFERENCE / "sklearn/neighbors/_classification.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
        batch = cases(components)
        if group == "gaussian-nb":
            assert Path(inspect.getfile(naive_bayes)).read_bytes() == (
                REFERENCE / "sklearn/naive_bayes.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
            batch += [
                dict(c, name=c["name"] + "-incremental", incremental=True)
                for c in batch if c["wrapper"] in ("ovr", "ovo")
            ]
        if group == "dummy":
            assert Path(inspect.getfile(DummyClassifier)).read_bytes() == (
                REFERENCE / "sklearn/dummy.py"
            ).read_bytes(), "install the pinned local scikit-learn checkout"
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
        if "pairwise_indices" in ref:
            checks["pairwise_indices"] = out["n_pairs"] == len(ref["pairwise_indices"]) and all(
                indices == out[f"pairwise_{pair}"]
                for pair, indices in enumerate(ref["pairwise_indices"])
            )
        for key in ("decision", "proba", "scores"):
            if key in ref and (key != "scores" or c["component"] in OUTPUT_CODE_RESPONSE):
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
