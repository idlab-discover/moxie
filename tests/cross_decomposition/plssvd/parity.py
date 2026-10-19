"""Differential parity of native PLSSVD against the pinned scikit-learn source.

Executes the unedited pinned `_pls.py` (via the PLS child's parity helpers,
which verify revision and cleanliness) and the compiled `builds/parity_runner`.
Tolerance: the reference and native both take one Float64 SVD of X.T @ y;
Jacobi vs LAPACK and summation order give ~1e-14 differences, so 1e-10 scaled
(1e-5 for ill-conditioned/tiny inputs) is used. Cases whose reference moves
beyond tolerance under a 1e-13 perturbation (repeated singular values, exact
sign ties) are compared by the singular values of the projected scores.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pls_parity", ROOT / "tests/cross_decomposition/pls/parity.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
RUNNER = ROOT / "builds/tests/cross_decomposition/plssvd/parity_runner"
OUT = ROOT / "builds/cross_decomposition/plssvd/parity"


def cases():
    out = []
    for name, X, y, loose in base.datasets():
        n, p = X.shape
        q = 1 if y.ndim == 1 else y.shape[1]
        rng = np.random.RandomState(zlib.crc32(name.encode()))
        m = max(2, n // 2)
        Xt = X[:m] + 0.1 * rng.randn(m, p)
        yt = y[:m] + 0.1 * rng.randn(*((m,) + y.shape[1:]))
        for k in sorted({1, min(2, n, p, q), min(n, p, q)}):
            for dtype in ("float64", "float32"):
                if dtype == "float32" and name not in ("linnerud", "tall-0", "wide", "univariate-y"):
                    continue
                for scale in (True, False):
                    out.append(dict(dtype=dtype, params=dict(n_components=k, scale=scale, copy=scale),
                                    X=X, y=y, Xt=Xt, yt=yt, loose=loose, data=name))
    rng = np.random.RandomState(9)
    Xe, ye = rng.randn(10, 5), rng.randn(10, 3)
    bad = [(dict(n_components=4), Xe, ye), (dict(n_components=0), Xe, ye), ({}, Xe[:1], ye[:1]),
           ({}, Xe, ye[:9]), ({}, np.where(np.arange(50).reshape(10, 5) == 3, np.nan, Xe), ye)]
    for i, (params, X, y) in enumerate(bad):
        full = dict(n_components=2, scale=True, copy=True)
        full.update(params)
        out.append(dict(dtype="float64", params=full, X=X, y=y, Xt=Xe, yt=ye, loose=False, data=f"invalid-{i}"))
    return out


def fixture_line(i, c):
    p = c["params"]
    cast = lambda a: np.asarray(a, dtype=c["dtype"])
    return "|".join([str(i), c["dtype"], str(p["n_components"]), "1" if p["scale"] else "0",
                     "1" if p["copy"] else "0", base.enc(cast(c["X"])), base.enc(cast(c["y"])),
                     base.enc(cast(c["Xt"])), base.enc(cast(c["yt"]))])


def reference(module, c, perturb=0.0):
    dtype = c["dtype"]
    X, y = np.asarray(c["X"], dtype=dtype), np.asarray(c["y"], dtype=dtype)
    Xt, yt = np.asarray(c["Xt"], dtype=dtype), np.asarray(c["yt"], dtype=dtype)
    if perturb:
        rng = np.random.RandomState(12345)
        X = X.astype(float) + perturb * X.std(axis=0) * rng.randn(*X.shape)
        y = y.astype(float) + perturb * y.std(axis=0) * rng.randn(*y.shape)
    est = module.PLSSVD(**{**c["params"], "copy": True})
    try:
        est.fit(X.copy(), y.copy())
    except Exception as exc:
        return {"error": (type(exc).__name__, str(exc))}
    xs, ys = est.transform(Xt, yt)
    Xw, yw = X.copy(), y.copy()
    module.PLSSVD(**c["params"]).fit(Xw, yw)
    return dict(x_weights_=est.x_weights_, y_weights_=est.y_weights_, transform=est.transform(Xt),
                transform_y=ys, names=",".join(est.get_feature_names_out()),
                fit_inplace_X=Xw, fit_inplace_y=yw)


def cross_rank(module, c):
    X, y = np.asarray(c["X"], dtype=float), np.asarray(c["y"], dtype=float)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    Xs, ys, *_ = module._center_scale_xy(X.copy(), y.copy(), c["params"]["scale"])
    s = np.linalg.svd(Xs.T @ ys, compute_uv=False)
    return int(np.sum(s > 1e-10 * s[0])) if s[0] > 0 else 0


def truncate(result, rank):
    out = dict(result)
    for key in ("x_weights_", "y_weights_", "transform", "transform_y"):
        out[key] = np.atleast_2d(result[key])[:, :rank]
    return out


def unstable(module, c, ref, tol):
    if "error" in ref:
        return False
    for k in range(ref["x_weights_"].shape[1]):
        top = np.sort(np.abs(ref["x_weights_"][:, k]))[::-1]
        if top.size > 1 and top[0] - top[1] <= 1e-10 * top[0]:
            return True
    other = reference(module, c, 1e-13)
    if "error" in other:
        return True
    for key in ("x_weights_", "y_weights_"):
        error, _ = base.array_error(ref[key], other[key], np.inf)
        if error is None or error > tol:
            return True
    return False


def main():
    base.verify_reference()
    module = base.load_reference()
    assert module.PLSSVD.__module__ == "pinned_pls"
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases = cases()
    fixture = OUT / "fixture.txt"
    fixture.write_text("\n".join(fixture_line(i, c) for i, c in enumerate(all_cases)) + "\n")
    completed = subprocess.run([RUNNER, fixture], capture_output=True, text=True, check=True)
    lines = completed.stdout.strip().split("\n")
    assert lines[-1] == "native|done|mojo", "native runner did not complete"
    got = {}
    for line in lines[:-1]:
        cid, key, value = line.split("|", 2)
        entry = got.setdefault(int(cid), {})
        entry[key] = value if key in ("error", "names") else base.dec(value)
    failures, rows, worst = [], [], 0.0
    for i, c in enumerate(all_cases):
        ref, mine = reference(module, c), got.get(i, {"error": "missing native result"})
        tol = base.LOOSE_TOL if c["loose"] else base.DEFAULT_TOL
        shaky = unstable(module, c, ref, tol)
        ok, message, error = True, "", 0.0
        if "error" in ref:
            kind, text = ref["error"]
            nk, _, nm_ = mine.get("error", "").partition(": ")
            kinds = {kind, "ValueError"} if kind == "InvalidParameterError" else {kind}
            ok = nk in kinds and base.first_line(nm_) == base.first_line(text)
            message = f"reference {kind}: {base.first_line(text)!r} native {mine.get('error')!r}"
        elif "error" in mine:
            ok, message = False, "native error: " + mine["error"]
        elif mine.get("names") != ref["names"]:
            ok, message = False, f"names {mine.get('names')} != {ref['names']}"
        else:
            rank = cross_rank(module, c)
            if rank < c["params"]["n_components"]:
                # Singular vectors for zero singular values are an arbitrary
                # orthonormal completion (LAPACK vs Jacobi pick different
                # ones): compare the leading `rank` components and require
                # the native completion to be orthonormal.
                ref = truncate(ref, rank)
                w = mine["x_weights_"]
                if not np.allclose(w.T @ w, np.eye(w.shape[1]), atol=1e-10):
                    ok, message = False, "native x_weights_ completion not orthonormal"
                mine = truncate(mine, rank)
            for key, expected in ref.items():
                if key == "names":
                    continue
                if shaky and key != "fit_inplace_X" and key != "fit_inplace_y":
                    # Compare the projected-score spectrum instead of vectors.
                    if key == "transform":
                        a = np.linalg.svd(np.atleast_2d(mine[key]), compute_uv=False)
                        b = np.linalg.svd(np.atleast_2d(expected), compute_uv=False)
                        e, good = base.array_error(b, a, 1e-6)
                    else:
                        e, good = base.array_error(expected, mine[key], np.inf)
                else:
                    e, good = base.array_error(expected, mine[key], tol)
                if e is None or not good:
                    ok, message = False, f"{key} error {e}"
                    break
                if not shaky:
                    error = max(error, e)
        label = f"{c['data']}/k={c['params']['n_components']}/{c['dtype']}/scale={c['params']['scale']}"
        rows.append(dict(case=label, ok=ok, unstable=shaky, worst_scaled_error=error, message=message))
        worst = max(worst, error) if ok else worst
        if not ok:
            failures.append(f"{label}: {message}")
    (OUT / "report.json").write_text(json.dumps(dict(
        reference=base.REVISION, runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        cases=rows), indent=1) + "\n")
    print(f"parity: {len(rows) - len(failures)}/{len(rows)} PLSSVD cases passed; worst scaled error {worst:.2e}; "
          f"{sum(r['unstable'] for r in rows)} rounding-sensitive cases compared by score spectra")
    for f in failures:
        print("FAIL", f)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
