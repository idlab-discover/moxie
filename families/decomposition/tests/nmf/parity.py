"""NMF, MiniBatchNMF and non_negative_factorization versus pinned sklearn/decomposition/_nmf.py.

The pinned module imports its Cython kernel _cdnmf_fast; the pinned .pyx is
compiled locally (tests/support/oracle.py) so the coordinate-descent code under
comparison is the pinned source too. Same seeds give the same NNDSVD/random
initialization and shuffles, so W and H are compared directly.
"""
import sys
import warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close, compile_pinned_extension

cdnmf = compile_pinned_extension("sklearn/decomposition/_cdnmf_fast.pyx")
module = load_pinned("sklearn/decomposition/_nmf.py", extensions={"_cdnmf_fast": cdnmf})
DRIVER = ROOT / "builds/nmf_driver"
count = 0


def code(k):
    return -1 if k == "auto" else -2 if k is None else k


def compare(x, kind="nmf", k="auto", init=None, solver="cd", beta="frobenius", tol=1e-4, max_iter=200, seed=0,
            alpha_W=0.0, alpha_H="same", l1_ratio=0.0, shuffle=False, batch_size=1024, forget=0.7, fresh=False,
            update_H=True, W0=None, H0=None, rtol=None):
    global count
    common = dict(init=init, beta_loss=beta, tol=tol, max_iter=max_iter, random_state=seed, alpha_W=alpha_W,
                  alpha_H=alpha_H, l1_ratio=l1_ratio)
    Wr = None if W0 is None else W0.copy()
    Hr = None if H0 is None else H0.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "function":
            W, H, n_iter = module.non_negative_factorization(x, Wr, Hr, k, solver=solver, update_H=update_H,
                                                               shuffle=shuffle, **common)
            expected = dict(W=W, H=H, n_iter=n_iter)
        elif kind == "nmf":
            ref = module.NMF(k, solver=solver, shuffle=shuffle, **common)
            W = ref.fit_transform(x, W=Wr, H=Hr)
            expected = dict(W=W, H=ref.components_, n_iter=ref.n_iter_, err=ref.reconstruction_err_, transform=ref.transform(x))
        else:
            ref = module.MiniBatchNMF(k, batch_size=batch_size, forget_factor=forget, fresh_restarts=fresh, **common)
            if kind == "partial":
                ref.partial_fit(x, W=Wr, H=Hr)
                ref.partial_fit(x)
                expected = dict(H=ref.components_, n_steps=ref.n_steps_, transform=ref.transform(x))
            else:
                W = ref.fit_transform(x, W=Wr, H=Hr)
                expected = dict(W=W, H=ref.components_, n_iter=ref.n_iter_, n_steps=ref.n_steps_,
                                err=ref.reconstruction_err_, transform=ref.transform(x))
    W0 = np.ones((1, 1)) if W0 is None else W0
    H0 = np.ones((1, 1)) if H0 is None else H0
    got = run_native(DRIVER, x.dtype, [x, W0, H0], kind, code(k), init or "none", solver, beta, tol, max_iter, seed,
                     alpha_W, -1 if alpha_H == "same" else alpha_H, l1_ratio, int(shuffle), batch_size, forget,
                     int(fresh), int(update_H))
    label = f"{x.dtype} {x.shape} {kind} {k=} {init} {solver} {beta} {tol=} {max_iter=} {alpha_W=} {alpha_H=} {l1_ratio=} {shuffle=} {batch_size=} {fresh=} {update_H=}"
    t = rtol or (1e-7 if x.dtype == np.float64 else 3e-3)
    for key in ("n_iter", "n_steps"):
        if key in expected:
            slack = 0 if x.dtype == np.float64 else 10
            assert abs(int(got[key][0]) - int(expected[key])) <= slack, (label, key, got[key], expected[key])
    scale = max(1.0, float(np.abs(x).max()))
    for key, value in expected.items():
        if key in ("n_iter", "n_steps"): continue
        assert_close(got.get(key), value, t, f"{label}: {key}", atol=t * scale)
    count += 1


rng = np.random.RandomState(9)
for dtype in (np.float64, np.float32):
    for shape, k in [((30, 8), 3), ((8, 20), 4), ((25, 6), 6)]:
        x = np.abs(rng.normal(size=shape)).astype(dtype)
        x[x < 0.2] = 0  # exact zeros exercise the divergence guards
        for init in (None, "random", "nndsvd", "nndsvda", "nndsvdar"):
            compare(x, k=k, init=init)
        compare(x, k=k, solver="cd", shuffle=True)
        compare(x, k=k, alpha_W=0.01, l1_ratio=0.5)
        compare(x, k=k, alpha_W=0.02, alpha_H=0.005, l1_ratio=0.0)
        compare(x, k=k, tol=0.0, max_iter=15)
        for beta in ("frobenius", "kullback-leibler", 1.5, 3.0):
            compare(x, k=k, solver="mu", beta=beta, init="nndsvda")
        compare(x, k=k, solver="mu", beta="kullback-leibler", alpha_W=0.01, l1_ratio=0.3, init="nndsvda")
        compare(x, k=None)
        W0 = np.abs(rng.normal(size=(shape[0], k))).astype(dtype)
        H0 = np.abs(rng.normal(size=(k, shape[1]))).astype(dtype)
        compare(x, k="auto", init="custom", W0=W0, H0=H0)
        compare(x, "function", k=k, init="custom", solver="mu", W0=W0, H0=H0)
        compare(x, "function", k="auto", update_H=False, H0=H0)
        compare(x, "function", k="auto", update_H=False, H0=H0, solver="mu")
        for batch in (7, 1024):
            compare(x, "minibatch", k=k, batch_size=batch, init="nndsvda")
        compare(x, "minibatch", k=k, batch_size=5, beta="kullback-leibler", init="nndsvda")
        compare(x, "minibatch", k=k, batch_size=5, fresh=True, init="nndsvda")
        compare(x, "minibatch", k=k, batch_size=6, alpha_W=0.01, l1_ratio=0.5, init="nndsvda")
        compare(x, "partial", k=k, batch_size=5, init="nndsvda")
    # Itakura-Saito needs strictly positive data.
    y = (np.abs(rng.normal(size=(20, 6))) + 0.1).astype(dtype)
    compare(y, k=3, solver="mu", beta="itakura-saito", init="nndsvda")
    compare(y, k=3, solver="mu", beta=0.5, init="nndsvda")
    compare(y, "minibatch", k=3, batch_size=6, beta="itakura-saito", init="nndsvda")
print(f"PASS: {count} dense NMF/MiniBatchNMF cases against local sklearn {REVISION}")
