"""Differential parity of the native PLS estimators against pinned scikit-learn.

The reference is the unedited `sklearn/cross_decomposition/_pls.py` from the
pinned local checkout, executed as its own module; the installed wheel only
supplies infrastructure (BaseEstimator, validation, r2_score). Python writes
fixtures and compares; all estimator arithmetic runs in the compiled Mojo
`builds/parity_runner` (its SHA-256 is recorded with the report).

Environment:
  MOXIE_WORKSPACE      workspace root (default: three levels above this child)
  MOXIE_PARITY_REPORT  report path (default: builds/parity/report.json)
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import warnings
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get("MOXIE_WORKSPACE", ROOT.parent)).resolve()
REFERENCE = WORKSPACE / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SOURCE = "sklearn/cross_decomposition/_pls.py"
RUNNER = ROOT / "builds/tests/cross_decomposition/pls/parity_runner"
OUT = ROOT / "builds/cross_decomposition/pls/parity"
REPORT = Path(os.environ.get("MOXIE_PARITY_REPORT", OUT / "report.json"))

# Tolerances: |native - reference| <= tol * max(1, max|reference|) + tol * |reference|.
# Both sides run the same Float64 algorithm; differences come only from
# summation order (BLAS vs scalar loops) and Jacobi vs LAPACK SVD. Measured on
# the well-conditioned cases the scaled error is <= 1.1e-13, and the reference
# itself moves by 2e-13..1e-12 under a 1e-13 input perturbation, so 1e-10
# leaves two orders of margin. `loose` cases are ill-conditioned by
# construction (nearly collinear blocks, n barely above k, CCA
# pseudoinverses); the reference moves ~1e-6 under the same perturbation, and
# 1e-5 is still far below any sign, branch or algorithm error (O(1)).
# Cases whose reference moves by more than its tolerance under that
# perturbation (degenerate canonical correlations, exact sign ties, rank
# below k) are compared by per-component objective values instead.
DEFAULT_TOL = 1e-10
LOOSE_TOL = 1e-5


def verify_reference():
    head = subprocess.check_output(["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True).strip()
    assert head == REVISION, f"reference drift: {head}"
    dirty = subprocess.check_output(["git", "-C", REFERENCE, "status", "--porcelain", "--", SOURCE], text=True)
    assert not dirty.strip(), f"reference source modified: {dirty}"


def load_reference():
    spec = importlib.util.spec_from_file_location("pinned_pls", REFERENCE / SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("PLSRegression", "PLSCanonical", "CCA"):
        cls = getattr(module, name)
        assert cls.__module__ == "pinned_pls"
        assert module._PLS.fit.__wrapped__.__code__.co_filename == str(REFERENCE / SOURCE)
        assert inspect.unwrap(module._PLS.transform).__code__.co_filename == str(REFERENCE / SOURCE)
    return module


def enc(a):
    if a is None:
        return "none"
    a = np.asarray(a)
    cols = a.shape[1] if a.ndim == 2 else 0
    return f"{a.shape[0]}:{cols}:" + ",".join(repr(float(v)) for v in a.ravel())


def dec(text):
    rows, cols, values = text.split(":", 2)
    rows, cols = int(rows), int(cols)
    data = np.array([float(v) for v in values.split(",")] if values else [], dtype=float)
    return data.reshape(rows, cols) if cols else data.reshape(rows)


def datasets():
    """(name, X, y, loose) tuples: tall/wide/tiny/ill-conditioned/degenerate."""
    from sklearn.datasets import load_linnerud, make_regression

    X, y = load_linnerud(return_X_y=True)
    yield "linnerud", X, y, False
    for seed in (0, 1, 2):
        rng = np.random.RandomState(seed)
        n, p, q = 40, 7, 3
        latent = rng.randn(n, 2)
        Xs = latent @ rng.randn(2, p) + 0.5 * rng.randn(n, p) + 3
        ys = latent @ rng.randn(2, q) + 0.3 * rng.randn(n, q) - 1
        yield f"tall-{seed}", Xs, ys, False
    rng = np.random.RandomState(7)
    yield "wide", rng.randn(12, 30), rng.randn(12, 4), False
    for seed in (530, 741):  # scikit-learn non-regression seeds for CCA (#18746)
        rng = np.random.RandomState(seed)
        yield f"tiny-{seed}", rng.randn(4, 3), rng.randn(4, 2), True
    Xc, yc = load_linnerud(return_X_y=True)
    Xc = Xc.copy()
    Xc[:, -1] = 1.0  # zero-std column
    yield "constant-x-column", Xc, yc, False
    rng = np.random.RandomState(3)
    base = rng.randn(30, 4)
    Xi = np.column_stack([base, base[:, 0] + 1e-7 * rng.randn(30)]) * 1000 + 50
    yield "ill-conditioned", Xi, rng.randn(30, 2), True
    rng = np.random.RandomState(4)
    yield "univariate-y", rng.randn(25, 5), rng.randn(25), False
    rng = np.random.RandomState(5)
    yield "two-samples", rng.randn(2, 3), rng.randn(2, 2), True
    X1, y1 = make_regression(n_samples=60, n_features=10, n_targets=5, random_state=11)
    yield "make-regression", X1, y1, False
    rng = np.random.RandomState(6)
    yc2 = rng.randn(20, 3)
    yc2[:, 1] = 2.5  # constant target column
    yield "constant-y-column", rng.randn(20, 4), yc2, False
    rng = np.random.RandomState(8)
    yield "constant-y", rng.rand(30, 3), np.zeros(30), False


def cases():
    specs = [
        ("PLSRegression", {}),
        ("PLSCanonical", {}),
        ("PLSCanonical", {"algorithm": "svd"}),
        ("CCA", {}),
    ]
    out = []
    for name, X, y, loose in datasets():
        n, p = X.shape
        q = 1 if y.ndim == 1 else y.shape[1]
        rng = np.random.RandomState(zlib.crc32(name.encode()))
        Xt = X[: max(2, n // 2)] + 0.1 * rng.randn(max(2, n // 2), p)
        yt = y[: max(2, n // 2)] + 0.1 * rng.randn(*((max(2, n // 2),) + y.shape[1:]))
        weights = rng.rand(Xt.shape[0]) + 0.1
        for estimator, extra in specs:
            bound = min(n, p) if estimator == "PLSRegression" else min(n, p, q)
            k = min(2, bound)
            for dtype in ("float64", "float32"):
                if dtype == "float32" and name not in ("linnerud", "tall-0", "wide", "univariate-y"):
                    continue
                for scale in (True, False):
                    if not scale and name not in ("linnerud", "tall-1", "ill-conditioned", "wide"):
                        continue
                    params = dict(n_components=k, scale=scale, max_iter=500, tol=1e-6, copy=True, **extra)
                    out.append(dict(estimator=estimator, dtype=dtype, params=params, X=X, y=y, Xt=Xt,
                                    yt=yt, w=weights if name == "tall-0" else None, loose=loose, data=name))
        # Full-rank decomposition and copy=False fit on each dataset for PLSRegression.
        out.append(dict(estimator="PLSRegression", dtype="float64",
                        params=dict(n_components=min(n, p), scale=True, max_iter=500, tol=1e-6, copy=False),
                        X=X, y=y, Xt=Xt, yt=yt, w=None, loose=True, data=name + "-full"))
    X, y = datasets_by_name("linnerud")
    out.append(dict(estimator="PLSCanonical", dtype="float64",
                    params=dict(n_components=3, scale=True, algorithm="nipals", max_iter=2, tol=1e-6, copy=True),
                    X=X, y=y, Xt=X, yt=y, w=None, loose=False, data="max-iter-2"))
    out.append(dict(estimator="CCA", dtype="float64",
                    params=dict(n_components=3, scale=True, max_iter=500, tol=1e-6, copy=False),
                    X=X, y=y, Xt=X, yt=y, w=None, loose=True, data="linnerud-cca-full"))
    from sklearn.datasets import make_regression
    X2, y2 = make_regression(n_samples=200, n_features=20, n_targets=20, random_state=0)
    out.append(dict(estimator="CCA", dtype="float64",
                    params=dict(n_components=10, scale=True, max_iter=500, tol=1e-6, copy=True),
                    X=X2, y=y2, Xt=X2[:20], yt=y2[:20], w=None, loose=True, data="cca-converges"))
    # Invalid inputs and parameters: native errors must match reference errors.
    rng = np.random.RandomState(9)
    Xe, ye = rng.randn(10, 5), rng.randn(10, 3)
    bad = [
        ("PLSRegression", dict(n_components=10), Xe, ye),
        ("PLSCanonical", dict(n_components=4), Xe, ye),
        ("CCA", dict(n_components=0), Xe, ye),
        ("PLSRegression", dict(max_iter=0), Xe, ye),
        ("PLSRegression", dict(tol=-1.0), Xe, ye),
        ("PLSCanonical", dict(algorithm="bad"), Xe, ye),
        ("PLSRegression", {}, Xe[:1], ye[:1]),
        ("PLSRegression", {}, Xe, ye[:9]),
        ("PLSRegression", {}, np.where(np.arange(50).reshape(10, 5) == 7, np.nan, Xe), ye),
        ("PLSRegression", {}, np.where(np.arange(50).reshape(10, 5) == 7, np.inf, Xe), ye),
        ("PLSRegression", {}, Xe, np.where(np.arange(30).reshape(10, 3) == 4, np.nan, ye)),
    ]
    for i, (estimator, params, X, y) in enumerate(bad):
        full = dict(n_components=2, scale=True, max_iter=500, tol=1e-6, copy=True)
        full.update(params)
        out.append(dict(estimator=estimator, dtype="float64", params=full, X=X, y=y, Xt=Xe, yt=ye,
                        w=None, loose=False, data=f"invalid-{i}"))
    return out


def datasets_by_name(target):
    for name, X, y, _ in datasets():
        if name == target:
            return X, y
    raise KeyError(target)


def fixture_line(i, case):
    p = case["params"]
    cast = (lambda a: None if a is None else np.asarray(a, dtype=case["dtype"]))
    return "|".join([
        str(i), case["estimator"], case["dtype"], str(p["n_components"]), "1" if p["scale"] else "0",
        p.get("algorithm", "nipals"), str(p["max_iter"]), repr(p["tol"]), "1" if p["copy"] else "0",
        enc(cast(case["X"])), enc(cast(case["y"])), enc(cast(case["Xt"])), enc(cast(case["yt"])),
        enc(case["w"]),
    ])


def reference(module, case, perturb=0.0):
    """Run the pinned estimator exactly as a scikit-learn user would.

    perturb > 0 adds perturb * column_std * N(0,1) noise (in Float64) to
    measure the reference's own sensitivity to rounding-level changes.
    """
    dtype = case["dtype"]
    X, y = np.asarray(case["X"], dtype=dtype), np.asarray(case["y"], dtype=dtype)
    if perturb:
        # Float64 additive noise scaled per column; constant columns stay
        # exactly constant so zero-std handling is not perturbed.
        rng = np.random.RandomState(12345)
        X, y = X.astype(np.float64), y.astype(np.float64)
        X = X + perturb * X.std(axis=0) * rng.randn(*X.shape)
        y = y + perturb * y.std(axis=0) * rng.randn(*y.shape)
    Xt, yt = np.asarray(case["Xt"], dtype=dtype), np.asarray(case["yt"], dtype=dtype)
    cls = getattr(module, case["estimator"])
    params = dict(case["params"])
    est = cls(**{**params, "copy": True})
    result = {"diagnostic": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            est.fit(X.copy(), y.copy())
        except Exception as exc:  # compared with the native error text
            return {"error": (type(exc).__name__, str(exc))}
    for w in caught:
        if issubclass(w.category, RuntimeWarning):
            continue  # NumPy floating-point warnings have no native analogue
        result["diagnostic"].append(f"{w.category.__name__}: {w.message}")
    for attr in ("x_weights_", "y_weights_", "x_loadings_", "y_loadings_", "x_rotations_",
                 "y_rotations_", "coef_", "intercept_"):
        result[attr] = getattr(est, attr)
    result["x_scores"], result["y_scores"] = est._x_scores, est._y_scores
    result["n_iter_"] = np.asarray(est.n_iter_, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["transform"] = est.transform(Xt.copy())
        xs, ys = est.transform(Xt.copy(), yt.copy())
        result["transform_y"] = ys
        result["predict"] = est.predict(Xt.copy())
        result["inverse_x"] = est.inverse_transform(xs)
        result["inverse_y"] = est.inverse_transform(xs, ys)[1]
        result["score"] = np.array([est.score(Xt, yt, sample_weight=case["w"])])
        result["names"] = ",".join(est.get_feature_names_out())
        Xc = Xt.copy()
        result["transform_inplace"] = est.transform(Xc, copy=False)
        result["transform_inplace_X"] = Xc
        Xp = Xt.copy()
        result["predict_inplace"] = est.predict(Xp, copy=False)
        result["predict_inplace_X"] = Xp
        Xw, yw = X.copy(), y.copy()
        cls(**params).fit(Xw, yw)
        result["fit_inplace_X"], result["fit_inplace_y"] = Xw, yw
    return result


def native(fixture):
    completed = subprocess.run([RUNNER, fixture], capture_output=True, text=True, check=True)
    lines = completed.stdout.strip().split("\n")
    assert lines[-1] == "native|done|mojo", "native runner did not complete"
    results = {}
    for line in lines[:-1]:
        case_id, key, value = line.split("|", 2)
        entry = results.setdefault(int(case_id), {"diagnostic": []})
        if key == "diagnostic":
            entry["diagnostic"].append(value)
        elif key in ("error", "names"):
            entry[key] = value
        else:
            entry[key] = dec(value)
    return results


def first_line(message):
    """First line of an error, with set displays sorted (Python set order is
    hash-randomized). scikit-learn appends a multi-line NaN imputation hint
    that refers to Python-only estimators; only its first line is compared."""
    line = message.split("\n")[0]
    if "{" in line and "}" in line:
        head, rest = line.split("{", 1)
        items, tail = rest.split("}", 1)
        line = head + "{" + ", ".join(sorted(i.strip() for i in items.split(","))) + "}" + tail
    return line


def objective(scores_x, scores_y, cca):
    """Per-component |corr| (CCA) or |cov| (PLS) of the paired training scores:
    the value each component maximizes, unique even when the vectors are not."""
    values = []
    for k in range(scores_x.shape[1]):
        t, u = scores_x[:, k] - scores_x[:, k].mean(), scores_y[:, k] - scores_y[:, k].mean()
        cov = abs(float(t @ u)) / max(len(t) - 1, 1)
        if cca:
            denom = float(np.sqrt((t @ t) * (u @ u))) / max(len(t) - 1, 1)
            cov = cov / denom if denom > 0 else 0.0
        values.append(cov)
    return np.array(values)


def array_error(expected, actual, tol):
    """Scaled error; None on shape/finiteness mismatch."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual) == np.isfinite(expected)):
        return None, False
    finite = np.isfinite(expected)
    if not finite.any():
        return 0.0, True
    scale = max(1.0, float(np.max(np.abs(expected[finite]))))
    error = np.abs(actual[finite] - expected[finite])
    if not np.isfinite(tol):
        return float(np.max(error) / scale), True
    return float(np.max(error) / scale), bool(np.all(error <= tol * scale + tol * np.abs(expected[finite])))


def compare(case, ref, got, unstable):
    """Return (ok, worst_scaled_error, message)."""
    if "error" in ref:
        kind, message = ref["error"]
        text = got.get("error", "")
        native_kind, _, native_message = text.partition(": ")
        if not case["data"].startswith("invalid"):
            # A numerical failure inside the reference fit (NaN from a zero
            # deflation denominator reaching the SVD): native must also fail.
            return bool(text), 0.0, f"reference {kind}: {message!r} native {text!r}"
        kinds = {kind, "ValueError"} if kind == "InvalidParameterError" else {kind}
        ok = native_kind in kinds and first_line(native_message) == first_line(message)
        return ok, 0.0, f"reference {kind}: {first_line(message)!r} native {text!r}"
    if "error" in got:
        return False, float("inf"), "native error: " + got["error"]
    tol = LOOSE_TOL if case["loose"] else DEFAULT_TOL
    worst = 0.0
    if got.get("names") != ref["names"]:
        return False, worst, f"feature names {got.get('names')} != {ref['names']}"
    if unstable:
        # The reference itself changes beyond tolerance under a 1e-13 relative
        # input perturbation, so its vectors are not a well-defined target.
        # Compare shapes, finiteness and the per-component objective values.
        for key, expected in ref.items():
            if key in ("diagnostic", "names", "n_iter_"):
                continue
            error, _ = array_error(expected, got[key], np.inf)
            if error is None:
                return False, worst, f"{key} shape/finiteness differs (unstable case)"
        cca = case["estimator"] == "CCA"
        error, ok = array_error(objective(ref["x_scores"], ref["y_scores"], cca),
                                objective(got["x_scores"], got["y_scores"], cca), 1e-6)
        return ok, error, "" if ok else f"component objectives differ by {error:.3e} (unstable case)"
    if got["diagnostic"] != ref["diagnostic"]:
        return False, worst, f"diagnostics {got['diagnostic']} != {ref['diagnostic']}"
    for key, expected in ref.items():
        if key in ("diagnostic", "names"):
            continue
        if key == "n_iter_":
            a, b = np.asarray(got[key]), np.asarray(expected)
            # Loose (tiny/ill-conditioned) cases may start NIPALS from a
            # rounding-noise y column (|values| just above 10*eps), which can
            # cost one extra iteration; the converged vectors are still compared.
            slack = 1 if case["loose"] else 0
            if a.shape != b.shape or np.any(np.abs(a - b) > slack):
                return False, worst, f"n_iter_ {a} != {b}"
            continue
        error, ok = array_error(expected, got[key], tol)
        if error is None:
            return False, worst, f"{key} shape/finiteness differs"
        worst = max(worst, error)
        if not ok:
            return False, worst, f"{key} scaled error {error:.3e} > {tol:.0e}"
    return True, worst, ""


def sensitivity(module, case, ref):
    """Scaled change of the reference's own fit under a 1e-13 perturbation."""
    if "error" in ref:
        return 0.0
    for k in range(ref["x_weights_"].shape[1]):
        # _svd_flip_1d picks the first largest |weight|; exact magnitude ties
        # are broken by rounding, so the component sign is not well defined.
        top = np.sort(np.abs(ref["x_weights_"][:, k]))[::-1]
        if top.size > 1 and top[0] > 0 and top[0] - top[1] <= 1e-10 * top[0]:
            return float("inf")
    other = reference(module, case, perturb=1e-13)
    if "error" in other:
        return float("inf")
    if not np.array_equal(np.asarray(other["n_iter_"]), np.asarray(ref["n_iter_"])):
        return float("inf")
    worst = 0.0
    for key in ("x_weights_", "y_weights_", "x_rotations_", "coef_"):
        error, _ = array_error(ref[key], other[key], np.inf)
        worst = max(worst, float("inf") if error is None else error)
    return worst


def main():
    verify_reference()
    module = load_reference()
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases = cases()
    fixture = OUT / "fixture.txt"
    fixture.write_text("\n".join(fixture_line(i, c) for i, c in enumerate(all_cases)) + "\n")
    got = native(fixture)
    failures, rows = [], []
    for i, case in enumerate(all_cases):
        ref = reference(module, case)
        drift = sensitivity(module, case, ref)
        tol = LOOSE_TOL if case["loose"] else DEFAULT_TOL
        unstable = drift > tol
        ok, worst, message = compare(case, ref, got.get(i, {"error": "missing native result", "diagnostic": []}), unstable)
        label = f"{case['data']}/{case['estimator']}{'/svd' if case['params'].get('algorithm') == 'svd' else ''}/{case['dtype']}/scale={case['params']['scale']}"
        rows.append(dict(case=label, ok=ok, unstable=unstable, reference_sensitivity=drift,
                         worst_relative_error=worst, message=message))
        if not ok:
            failures.append(f"{label}: {message}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(dict(
        reference=REVISION, source=SOURCE, runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        default_tol=DEFAULT_TOL, loose_tol=LOOSE_TOL, cases=rows), indent=1) + "\n")
    worst = max((r["worst_relative_error"] for r in rows if r["ok"]), default=0.0)
    unstable = [r["case"] for r in rows if r["unstable"]]
    print(f"parity: {len(rows) - len(failures)}/{len(rows)} cases passed; worst passing scaled error {worst:.2e}")
    print(f"parity: {len(unstable)} rounding-sensitive reference cases compared by component objectives:")
    for name in unstable:
        print("  unstable", name)
    for f in failures:
        print("FAIL", f)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
