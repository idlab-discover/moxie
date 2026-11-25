"""Compare linear-kernel SVC selectors to pinned sklearn and pinned LIBSVM."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "builds/svc-importance-parity"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "families/linear_model/tools"))
from compiler_limits import compiler_command
import parity_io as oracle

HEADER = """\
import numojo as nm
from std.testing import assert_equal
from moxie.feature_selection import (
    DenseSupervisedCV, FeatureCount, RFE, RFECV, SVCImportance,
    SelectFromModel, SequentialCV,
)
from moxie.svm import SVC

def dump[dtype: DType](values: nm.NDArray[dtype]) raises:
    print(values.ndim)
    for axis in range(values.ndim):
        print(values.shape[axis])
    for i in range(values.size):
        print(values.unsafe_get(i))
"""


def pinned_estimators():
    base, model, elimination = [
        f"sklearn/feature_selection/{name}.py" for name in ("_base", "_from_model", "_rfe")
    ]
    oracle.verify_reference([base, model, elimination])
    common = oracle.load_pinned(base, register=True)
    from_model = oracle.load_pinned(model)
    rfe = oracle.load_pinned(elimination)
    assert from_model._get_feature_importances is common._get_feature_importances
    assert rfe._get_feature_importances is common._get_feature_importances

    # Reuse the SVM owner's bridge that builds and executes the pinned C++ engine.
    path = ROOT / "families/svm/tests/public_kernel_parity.py"
    spec = importlib.util.spec_from_file_location("pinned_kernel_estimators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    svc = module.reference_classes()["SVC"]
    oracle.pinned_code(svc.fit, "sklearn/svm/_base.py")
    return svc, (from_model.SelectFromModel, rfe.RFE, rfe.RFECV)


def fixture(dtype, labels, selector_type, svc):
    rng = np.random.default_rng(118)
    groups = rng.permutation(np.tile(np.arange(len(labels)), 12))
    X = rng.normal(scale=0.22, size=(len(groups), 4))
    X[:, 2] += 2.7 * groups
    X[:, 0] += 1.2 * (groups != 0)
    X = X.astype(dtype)
    y = np.array(labels)[groups]
    weights = np.linspace(0.5, 1.5, len(y)).astype(dtype)
    name = selector_type.__name__
    weighted = name != "RFECV"
    getter = "auto" if name == "SelectFromModel" else "coef_"
    options = {"RFE": {"n_features_to_select": 2}, "RFECV": {"cv": 3}}.get(name, {})
    model = svc(kernel="linear", C=0.7, tol=1e-7, random_state=11, probability=True)
    reference = selector_type(model, importance_getter=getter, **options)
    reference.fit(X, y, **({"sample_weight": weights} if weighted else {}))

    label = "String" if isinstance(labels[0], str) else "Int64"
    lines = [HEADER, f"comptime dtype = DType.{np.dtype(dtype).name}",
             f"comptime E = SVCImportance[{label}, dtype]", "def main() raises:"]
    numbers = []
    exact_count = 0

    def add(line):
        lines.append("    " + line)

    def exact(expression, expected):
        nonlocal exact_count
        # JSON and Mojo share list syntax but spell boolean literals differently.
        literal = json.dumps(np.asarray(expected).tolist(), ensure_ascii=False)
        if np.asarray(expected).dtype == np.bool_:
            literal = literal.replace("true", "True").replace("false", "False")
        add(f"assert_equal({expression}, {literal})")
        exact_count += np.size(expected)

    def dump(expression, expected, expected_dtype=None):
        values = np.asarray(expected)
        if expected_dtype is not None:
            add(f"comptime assert type_of({expression}) == nm.NDArray[{expected_dtype}]")
        add(f"dump({expression})")
        numbers.extend([values.ndim, *values.shape, *values.flat])

    def emit_strided(name, values):
        reversed_values = ", ".join(repr(float(value)) for value in values.ravel()[::-1])
        # Leading unused element keeps the live view at a nonzero offset.
        add(f"var {name}_backing = nm.array[dtype]([0.0, {reversed_values}], [{values.size + 1}])")
        shape = ", ".join(map(str, values.shape))
        strides = f"-{values.shape[1]}, -1" if values.ndim == 2 else "-1"
        add(f"var {name} = {name}_backing.view_with_layout(nm.Shape({shape}), "
            f"nm.NDArrayStrides({strides}), {values.size})")

    emit_strided("X", X)
    add(f"var y: List[{label}] = {json.dumps(y.tolist(), ensure_ascii=False)}")
    if weighted:
        emit_strided("weights", weights)
    add(f'var estimator = E(SVC[{label}](kernel="linear", C=0.7, tol=1e-7, '
        'random_state=11, probability=True))')
    constructors = {
        "SelectFromModel": "SelectFromModel(estimator^)",
        "RFE": 'RFE(estimator^, n_features_to_select=FeatureCount(2), importance_getter="coef_")',
        "RFECV": 'RFECV[DenseSupervisedCV[E, dtype]](estimator^, cv=SequentialCV(3), importance_getter="coef_")',
    }
    add(f"var selector = {constructors[name]}")
    target = "y.copy()" if name == "SelectFromModel" else "y"
    add(f'_ = selector.fit(X, {target}{", sample_weight=weights" if weighted else ""})')
    exact("selector.get_support()[List[Bool]]", reference.get_support())
    dump("selector.transform(X)", reference.transform(X), "dtype")
    add("var fitted = selector.estimator_()")
    exact("fitted.classes_()", reference.estimator_.classes_)
    # Read outputs before the next sklearn fit replaces the bridge's native state.
    dump('fitted.importances("coef_")', reference.estimator_.coef_.astype(dtype), "dtype")
    component = "fitted" if name == "SelectFromModel" else "selector"
    predictor = reference.estimator_ if component == "fitted" else reference
    exact(f"{component}.predict(X)", predictor.predict(X))
    dump(f"{component}.decision_function(X)", predictor.decision_function(X), "DType.float64")
    dump(f"{component}.predict_proba(X)", predictor.predict_proba(X), "DType.float64")
    dump(f"{component}.predict_log_proba(X)", predictor.predict_log_proba(X), "DType.float64")
    add(f'print({component}.score(X, y{", weights.copy()" if weighted else ""}))')
    numbers.append(predictor.score(X, y, **({"sample_weight": weights} if weighted else {})))
    if name == "SelectFromModel":
        add("print(selector.threshold_())")
        numbers.append(reference.threshold_)
    else:
        exact("selector.ranking_()", reference.ranking_)
    if name == "RFECV":
        add("var results = selector.cv_results_()")
        for key, values in reference.cv_results_.items():
            if key == "n_features":
                exact(f'results["{key}"][List[Int]]', values)
            elif key.endswith(("_support", "_ranking")):
                typ = "Bool" if key.endswith("_support") else "Int"
                add(f'var {key} = results["{key}"][List[List[{typ}]]].copy()')
                for row, entries in enumerate(values):
                    exact(f"{key}[{row}]", entries)
            else:
                dump(f'results["{key}"][nm.NDArray[DType.float64]]', values)
    return "\n".join(lines) + "\n", numbers, exact_count


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    svc, selector_types = pinned_estimators()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for selector_type in selector_types:
        for dtype, labels in ((np.float64, [2**53 + 1, 2**53]), (np.float32, ["é", "alpha", "z"])):
            code, expected, comparisons = fixture(dtype, labels, selector_type, svc)
            source = OUTPUT / f"{selector_type.__name__}-{np.dtype(dtype).name}.mojo"
            source.write_text(code)
            binary = source.with_suffix("")
            log = source.with_suffix(".log")
            with log.open("w") as output:
                command = compiler_command(["mojo", "build", "-O3", "--fp-mode=contract=off",
                                            "-I", ROOT / "builds/deps", source, "-o", binary])
                if subprocess.run(command, stdout=output, stderr=output).returncode:
                    raise RuntimeError(f"compiler log at {log}")
            actual = np.array(subprocess.check_output([binary], text=True).split(), dtype=float)
            # Use the kernel owner's tolerances for solver and calibration output.
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5, err_msg=source.stem)
            count += comparisons + len(expected)
            print(f"PASS {source.stem} ({comparisons + len(expected)} comparisons)", flush=True)
    print(f"PASS linear-kernel SVC selector parity ({count} comparisons)")


if __name__ == "__main__":
    main()
