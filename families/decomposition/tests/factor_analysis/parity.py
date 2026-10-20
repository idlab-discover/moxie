"""FactorAnalysis versus the pinned local sklearn/decomposition/_factor_analysis.py.

The reference keeps LAPACK's unflipped singular-vector signs for
svd_method='lapack', so component rows and the matching latent columns are
compared up to one sign per factor. Randomized fits (U-based svd_flip) and
rotated fits are compared the same way because a rotation of sign-flipped
loadings is the sign-flipped rotation. Noise variances, the log-likelihood
trace, covariance, precision and scores are sign-invariant and compared directly.
Both implementations run EM in Float64 (the reference casts Float32 input).
"""
import sys
import warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close, align_signs

module = load_pinned("sklearn/decomposition/_factor_analysis.py")
Reference = module.FactorAnalysis
DRIVER = ROOT / "builds/factor_analysis_driver"
# EM compounds the rounding of two different SVD kernels over many sweeps; the
# fixed point is well conditioned, so a 1e-7 relative/absolute bound (scaled by
# the data magnitude for data-unit quantities) still detects any formula change.
TOL = 1e-7
count = 0


def compare(x, k, method="randomized", rotation=None, max_iter=1000, tol=1e-2, power=3, seed=0, psi=None):
    global count
    reference = Reference(k, svd_method=method, rotation=rotation, max_iter=max_iter, tol=tol,
                          iterated_power=power, random_state=seed, noise_variance_init=psi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_z = reference.fit_transform(x)
    psi_matrix = np.ones((1, x.shape[1])) if psi is None else np.asarray(psi, dtype=np.float64)[None, :]
    got = run_native(DRIVER, x.dtype, [x, psi_matrix], -1 if k is None else k, method, rotation or "none",
                     max_iter, tol, power, seed, int(psi is not None))
    label = f"{x.dtype} {x.shape} {k=} {method} {rotation} {max_iter=} {tol=} {power=}"
    assert int(got["n_iter"][0]) == reference.n_iter_, (label, got["n_iter"], reference.n_iter_)
    scale = max(1.0, float(np.abs(x).max()))
    assert_close(got["loglike"], reference.loglike_, TOL, label + ": loglike", atol=TOL * x.size * scale)
    for key, value in [("noise", reference.noise_variance_), ("mean", reference.mean_),
                       ("covariance", reference.get_covariance())]:
        assert_close(got[key], value, TOL, f"{label}: {key}", atol=TOL * scale ** 2)
    assert_close(got["precision"], reference.get_precision(), TOL * 10, label + ": precision", atol=TOL * 10 / min(1.0, reference.noise_variance_.min()))
    assert_close(got["score_samples"], reference.score_samples(x), TOL, label + ": score_samples", atol=TOL * 10 * x.shape[1])
    assert_close(got.get("score"), reference.score(x), TOL, label + ": score", atol=TOL * 10 * x.shape[1])
    kk = reference.components_.shape[0]
    if kk:
        components, signs = align_signs(got["components"].reshape(kk, x.shape[1]), reference.components_, 0)
        assert_close(components, reference.components_, TOL, label + ": components", atol=TOL * scale)
        latent = got["transform"].reshape(x.shape[0], kk) * signs[None, :]
        assert_close(latent, expected_z, TOL * 10, label + ": transform", atol=TOL * 10)
    else:
        assert "components" not in got and "transform" not in got
    count += 1


rng = np.random.RandomState(4242)
for dtype in (np.float64, np.float32):
    for n, p in [(40, 6), (25, 3), (10, 12), (60, 10)]:
        latent = rng.normal(size=(n, 2))
        loadings = rng.normal(size=(2, p)) * 2
        x = (latent @ loadings + rng.normal(size=(n, p)) * rng.uniform(0.2, 1.5, size=p)).astype(dtype)
        for method in ("lapack", "randomized"):
            # Centered data has rank <= n - 1: with n <= p, k=None retains a zero
            # singular value whose log dominates the likelihood with pure rounding
            # noise in both implementations, so it is not a meaningful comparison.
            for k in ((None,) if n > p else ()) + (0, 1, 2, min(3, p)):
                compare(x, k, method)
            for rotation in ("varimax", "quartimax"):
                compare(x, 2, method, rotation)
            compare(x, 2, method, tol=1e-6)
            compare(x, 2, method, max_iter=3)
            compare(x, 2, method, psi=rng.uniform(0.5, 2, size=p))
        for power in (0, 1, 5):
            compare(x, 2, "randomized", power=power, seed=int(rng.randint(1000)))
    # Constant feature (zero variance hits the SMALL floor) and a scaled problem.
    x = rng.normal(size=(30, 4)).astype(dtype)
    x[:, 2] = 3.0
    compare(x, 2, "lapack")
    compare(x, 2, "randomized")
    compare((rng.normal(size=(30, 5)) * 1e3).astype(dtype), 2, "lapack")
print(f"PASS: {count} dense FactorAnalysis cases against local sklearn {REVISION}")
