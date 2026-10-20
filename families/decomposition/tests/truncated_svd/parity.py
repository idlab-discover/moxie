"""TruncatedSVD versus the pinned local sklearn/decomposition/_truncated_svd.py."""
import sys
import warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close, tolerance

module = load_pinned("sklearn/decomposition/_truncated_svd.py")
Reference = module.TruncatedSVD
DRIVER = ROOT / "builds/truncated_svd_driver"
count = 0


def compare(x, k, algorithm="randomized", n_iter=5, oversamples=10, normalizer="auto", seed=0, tol=0.0, unique=True):
    global count
    reference = Reference(k, algorithm=algorithm, n_iter=n_iter, n_oversamples=oversamples,
                          power_iteration_normalizer=normalizer, random_state=seed, tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_fit = reference.fit_transform(x)
        expected_transform = reference.transform(x)
        expected_inverse = reference.inverse_transform(expected_transform)
    got = run_native(DRIVER, x.dtype, [x], k, algorithm, n_iter, oversamples, normalizer, seed, tol)
    t = tolerance(x.dtype)
    # Randomized factors pass through products and QR/LU of the sketch; allow
    # a multiple of the direct-factor tolerance for accumulated rounding.
    if algorithm == "randomized": t *= 30
    scale = max(1.0, np.abs(x).max())
    label = f"{x.dtype} {x.shape} {k=} {algorithm} {n_iter=} {normalizer} {tol=}"
    for key, value in [("singular", reference.singular_values_), ("variance", reference.explained_variance_),
                       ("ratio", reference.explained_variance_ratio_), ("inverse", expected_inverse)]:
        assert_close(got.get(key), value, t, f"{label}: {key}", atol=t * scale ** 2 if key == "variance" else t * scale)
    if unique:
        for key, value in [("components", reference.components_), ("fit_transform", expected_fit),
                           ("transform", expected_transform)]:
            assert_close(got.get(key), value, t, f"{label}: {key}", atol=t * scale)
    else:
        # Directions with (numerically) zero singular value are arbitrary null-space
        # vectors; compare the projector onto the nonzero singular subspace only.
        kk = reference.components_.shape[0]
        keep = reference.singular_values_ > reference.singular_values_.max() * np.finfo(x.dtype).eps * 100
        c = got["components"].reshape(kk, x.shape[1])[keep]
        r = reference.components_[keep]
        assert_close(c.T @ c, r.T @ r, t, f"{label}: subspace")
    count += 1


rng = np.random.RandomState(1213)
for dtype in (np.float64, np.float32):
    for shape in [(30, 8), (8, 30), (12, 5), (60, 4)]:
        x = rng.normal(size=shape).astype(dtype)
        for k in (1, 2, min(shape) - 1):
            compare(x, k, "arpack")
            compare(x, k, "arpack", tol=1e-8)
            for normalizer in ("auto", "none", "QR", "LU"):
                for n_iter in (0, 3, 5):
                    compare(x, k, "randomized", n_iter=n_iter, normalizer=normalizer, seed=int(rng.randint(100)))
        compare(x, min(shape), "randomized")
    # Nonnegative count-like data (the LSA use case) and a wide oversampled sketch.
    counts = rng.poisson(1.5, size=(40, 12)).astype(dtype)
    for k in (2, 5):
        compare(counts, k, "arpack")
        compare(counts, k, "randomized", oversamples=3)
    # Rank-deficient input: only the nonzero subspace is unique.
    low = (rng.normal(size=(20, 2)) @ rng.normal(size=(2, 6))).astype(dtype)
    compare(low, 2, "arpack")
    compare(low, 2, "randomized", n_iter=7)
    compare(low, 3, "arpack", unique=False)
print(f"PASS: {count} dense TruncatedSVD cases against local sklearn {REVISION}")
