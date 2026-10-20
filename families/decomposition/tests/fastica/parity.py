"""FastICA versus the pinned local sklearn/decomposition/_fastica.py.

Both implementations draw w_init from the same NumPy-compatible RandomState and
use sign-normalized whitening (u *= sign(u[0])), and symmetric decorrelation is
basis invariant, so estimated sources, unmixing and mixing matrices are
compared directly, without permutation or sign alignment.
"""
import sys
import warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close

module = load_pinned("sklearn/decomposition/_fastica.py")
Reference = module.FastICA
DRIVER = ROOT / "builds/fastica_driver"
count = 0


def same_iterations(got, expected, dtype, label):
    # Float32 stopping tests compare a rounding-level change against tol, so the
    # final sweep may land one or two iterations apart; Float64 must be exact.
    slack = 0 if dtype == np.float64 else 2
    assert abs(int(got) - int(expected)) <= slack, (label, got, expected)


def tol_for(dtype, iterative):
    # Fixed-point iterations amplify rounding of different factorization
    # kernels; Float32 additionally follows float32 NumPy arithmetic.
    base = 1e-7 if dtype == np.float64 else 2e-3
    return base * (10 if iterative else 1)


def compare(x, k=None, algorithm="parallel", whiten="unit-variance", fun="logcosh", alpha=1.0,
            max_iter=200, tol=1e-4, solver="svd", seed=0, w_init=None, mode="fit_transform"):
    global count
    kwargs = dict(n_components=k, algorithm=algorithm, whiten=whiten, fun=fun,
                  fun_args=None if alpha == 1.0 else {"alpha": alpha}, max_iter=max_iter, tol=tol,
                  w_init=w_init, whiten_solver=solver, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "function":
            expected = module.fastica(x, return_X_mean=True, return_n_iter=True, **kwargs)
        else:
            reference = Reference(**kwargs)
            sources = reference.fit_transform(x) if mode == "fit_transform" else None
            if mode == "fit": reference.fit(x)
            transform = reference.transform(x)
            inverse = reference.inverse_transform(transform)
    w0 = np.eye(1) if w_init is None else w_init
    got = run_native(DRIVER, x.dtype, [x, w0], -1 if k is None else k, algorithm,
                     "false" if whiten is False else whiten, fun, alpha, max_iter, tol, solver, seed,
                     int(w_init is not None), mode)
    label = f"{x.dtype} {x.shape} {k=} {algorithm} {whiten} {fun} {alpha=} {solver} {mode}"
    t = tol_for(x.dtype, True)
    if mode == "function":
        K, W, S, mean, n_iter = expected
        same_iterations(got["n_iter"][0], n_iter, x.dtype, label)
        if K is not None:
            assert_close(got["K"], K, t, label + ": K")
            assert_close(got["mean"], mean, t, label + ": mean")
        assert_close(got["W"], W, t, label + ": W")
        assert_close(got["sources"], S, t, label + ": S")
    else:
        same_iterations(got["n_iter"][0], reference.n_iter_, x.dtype, label)
        pairs = [("components", reference.components_), ("mixing", reference.mixing_),
                 ("transform", transform), ("inverse", inverse)]
        if sources is not None: pairs.append(("sources", sources))
        if whiten is not False:
            pairs += [("mean", reference.mean_), ("whitening", reference.whitening_)]
        scale = max(1.0, float(np.abs(x).max()))
        for key, value in pairs:
            assert_close(got[key], value, t, f"{label}: {key}", atol=t * scale)
    count += 1


rng = np.random.RandomState(515)
for dtype in (np.float64, np.float32):
    for n in (200, 60):
        t = np.linspace(0, 8, n)
        # Four non-Gaussian sources: a Gaussian noise channel would make the
        # k=None solution ill-posed (ICA cannot identify Gaussian directions).
        sources = np.column_stack([np.sin(2 * t), np.sign(np.sin(3 * t)), rng.laplace(size=n),
                                   (t * 1.7) % 1.0 - 0.5])
        mixing = rng.normal(size=(4, 4))
        x = (sources @ mixing).astype(dtype)
        for algorithm in ("parallel", "deflation"):
            for fun in ("logcosh", "exp", "cube"):
                for whiten in ("unit-variance", "arbitrary-variance"):
                    for k in (None, 3):
                        compare(x, k, algorithm, whiten, fun, seed=int(rng.randint(1000)))
                compare(x, 2, algorithm, fun=fun, solver="eigh")
            compare(x, 3, algorithm, alpha=1.5)
            compare(x, 3, algorithm, max_iter=3)
            # tol=1e-8 is below Float32 rounding: its stopping step is noise-driven.
            if dtype == np.float64: compare(x, 3, algorithm, tol=1e-8, max_iter=400)
            compare(x, 3, algorithm, w_init=rng.normal(size=(3, 3)))
            compare(x, 3, algorithm, mode="fit")
            compare(x, 3, algorithm, mode="function")
            compare(x, 10, algorithm)  # clipped to min(n, p)
        # whiten=False works on the raw features (square unmixing).
        z = (sources[:, :3] @ rng.normal(size=(3, 3))).astype(dtype)
        compare(z, None, "parallel", whiten=False, max_iter=500)
        compare(z, None, "deflation", whiten=False, max_iter=500)
print(f"PASS: {count} dense FastICA cases against local sklearn {REVISION}")
