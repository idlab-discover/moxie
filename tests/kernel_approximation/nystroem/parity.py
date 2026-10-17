"""Differential comparison with the pinned sklearn Nystroem and pairwise kernel bodies.

Exact: component_indices_ and the next RNG draw (permutation consumption).
Numerical: normalization_ = U diag(clip(S,1e-12))^-1/2 V^T is a conditioned
matrix function of the basis Gram. With kappa = s_max / max(s_min, 1e-12) of
the reference spectrum, both normalization_ and transform are compared with
|native - sklearn| <= 64 * eps(dtype) * kappa * max(1, max|sklearn|).
Cases with kappa above 1e6 (Float64) / 1e2 (Float32) are ill-conditioned: there
the entries are not determined to working precision (the SVD bases differ:
native Jacobi vs LAPACK gesdd), so the basis-invariant property of sklearn's
test_nystroem_singular_kernel is checked instead: for Float64,
Z_B Z_B^T = U S U^T (|K_BB|, basis independent) must agree with sklearn's;
for Float32 with kappa <= 1e4 the same within 1e-3 of max|K_BB|, beyond that
check finiteness and held-out transform composition with kernel/dot-product
roundoff bounds. Independent analytic spectra additionally test actual feature
Gram accuracy above, at and below the singular-value floor in both dtypes.
"""
import json, subprocess, tempfile, warnings
from pathlib import Path
import numpy as np
from joblib import effective_n_jobs
from oracle import load, ROOT, REV
Nystroem, pairwise = load()
count = 0
worst = {'float32': 0.0, 'float64': 0.0}
ill = []


def fmt(v):
    return 'none' if v is None else repr(float(v))


def native(kernel, x, y, gamma, coef0, degree, m, seed, n_jobs=1):
    bits = '32' if x.dtype == np.float32 else '64'
    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir=ROOT / 'builds', delete=False) as f:
        f.write(f'{bits} {kernel} {fmt(gamma)} {fmt(coef0)} {fmt(degree)} {m} {seed} {x.shape[0]} {x.shape[1]}\n')
        f.write(' '.join(repr(float(v)) for v in x.ravel()))
        f.write(f'\n{y.shape[0]}\n' + ' '.join(repr(float(v)) for v in y.ravel()) + f'\n{n_jobs}\n')
    result = subprocess.run([ROOT / 'builds/tests/kernel_approximation/nystroem/evaluate', f.name], capture_output=True, text=True)
    Path(f.name).unlink()
    return result


def compare(kernel, x, y, gamma=None, coef0=None, degree=None, m=5, seed=0, n_jobs=1):
    global count
    label = (kernel, x.dtype.name, x.shape, gamma, coef0, degree, m, seed, n_jobs)
    rng = np.random.RandomState(seed)
    reference = Nystroem(kernel, gamma=gamma, coef0=coef0, degree=degree, n_components=m, random_state=rng, n_jobs=n_jobs)
    result = native(kernel, x, y, gamma, coef0, degree, m, seed, n_jobs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reference.fit(x)
            want = reference.transform(y)
    except ValueError as error:
        assert result.returncode != 0 and 'ValueError' in result.stderr + result.stdout, (label, error, result.stdout)
        count += 1
        return
    assert result.returncode == 0, (label, result.stderr)
    values = result.stdout.split()
    k = reference.component_indices_.size
    assert np.array_equal(np.array(values[:k], dtype=np.int64), reference.component_indices_), label
    assert float(values[-1]) == rng.random_sample(), (label, 'draw consumption')
    norm = np.array(values[k:k + k * k], dtype=float).reshape(k, k)
    got = np.array(values[k + k * k:-1], dtype=float).reshape(want.shape)
    # The native driver explicitly selects Float64 output for Float32 Laplacian.
    expected_dtype = np.float64 if kernel == 'laplacian' and effective_n_jobs(n_jobs) == 1 else x.dtype
    assert want.dtype == expected_dtype and reference.normalization_.dtype == expected_dtype, label
    basis_kernel = pairwise['pairwise_kernels'](reference.components_.astype(np.float64), metric=kernel,
                                                filter_params=True, **reference._get_kernel_params())
    s = np.linalg.svd(basis_kernel, compute_uv=False)
    kappa = s[0] / max(s[-1], 1e-12) if s[0] > 0 else np.inf  # zero Gram: any orthogonal U V^T is valid
    eps = np.finfo(expected_dtype).eps
    # Test actual held-out transform arithmetic even when an entrywise SVD
    # comparison is ill-conditioned. A dot product has gamma_k error scaled
    # by sum(abs(products)); the kernel comparison is covered separately.
    cross = pairwise['pairwise_kernels'](y, reference.components_, metric=kernel,
                                       filter_params=True, **reference._get_kernel_params())
    composed = cross.astype(float) @ norm.T
    roundoff = 8 * k * eps * np.maximum(1, np.abs(cross.astype(float)) @ np.abs(norm.T))
    # Shared Float32 kernel arithmetic can differ by a few ulps from BLAS.
    kernel_roundoff = 32 * x.shape[1] * np.finfo(x.dtype).eps * np.maximum(1, np.abs(cross)) @ np.abs(norm.T)
    assert np.all(np.abs(got - composed) <= roundoff + kernel_roundoff), (label, 'transform composition')
    if kappa <= (1e6 if expected_dtype == np.float64 else 1e2):
        for name, a, b in (('normalization_', norm, reference.normalization_), ('transform', got, want)):
            bound = 64 * eps * kappa * max(1.0, float(np.max(np.abs(b))))
            error = float(np.max(np.abs(a - b)))
            assert error <= bound, (label, name, error, bound, kappa)
            worst[x.dtype.name] = max(worst[x.dtype.name], error / bound)
    else:
        assert np.all(np.isfinite(got)), label
        moderately = expected_dtype == np.float32 and kappa <= 1e4
        if (expected_dtype == np.float64 or moderately) and kernel != 'precomputed':
            # Z_B Z_B^T = K N^T N K = U S U^T (= |K_BB|) is independent of the SVD bases.
            zb = basis_kernel @ norm.T
            zr = basis_kernel @ reference.normalization_.T
            gram_error = float(np.max(np.abs(zb @ zb.T - zr @ zr.T)))
            tol = 1e-6 if expected_dtype == np.float64 else 1e-3
            assert gram_error <= tol * max(1.0, np.max(np.abs(basis_kernel))), (label, gram_error)
            ill.append(dict(case=str(label), kappa=float(kappa), basis_gram_error=gram_error))
        else:
            ill.append(dict(case=str(label), kappa=float(kappa), checked='finite and held-out transform composition'))
    count += 1


r = np.random.RandomState(0)
for dtype in (np.float32, np.float64):
    tall = r.uniform(0, 1, (40, 5)).astype(dtype); tall_y = r.uniform(0, 1, (7, 5)).astype(dtype)
    wide = r.uniform(0, 1, (6, 20)).astype(dtype); wide_y = r.uniform(0, 1, (3, 20)).astype(dtype)
    tiny = np.array([[0.4]], dtype=dtype)
    duplicated = np.repeat(r.uniform(0, 1, (4, 3)), 3, axis=0).astype(dtype)
    settings = [('rbf', {}), ('rbf', dict(gamma=0.5)), ('linear', {}), ('poly', {}),
                ('poly', dict(gamma=0.3, coef0=0.5, degree=2)), ('polynomial', dict(degree=3.0)),
                ('sigmoid', dict(gamma=0.05, coef0=0.1)), ('laplacian', dict(gamma=0.7)), ('cosine', {}),
                ('chi2', {}), ('chi2', dict(gamma=0.3)), ('additive_chi2', {})]
    for kernel, kw in settings:
        for seed, m in ((0, 5), (3, 12)):
            compare(kernel, tall, tall_y, m=m, seed=seed, **kw)
        compare(kernel, wide, wide_y, m=100, seed=1, **kw)  # n_components > n_samples
        compare(kernel, tiny, tiny, m=3, seed=2, **kw)
        compare(kernel, duplicated, duplicated[:4], m=10, seed=4, **kw)  # rank-deficient basis
    compare('rbf', tall, tall_y, m=5, seed=4294967295)
    # Precomputed: full basis required (k == n), else both implementations raise.
    gram = pairwise['rbf_kernel'](tall[:10].astype(np.float64)).astype(dtype)
    cross = pairwise['rbf_kernel'](tall_y.astype(np.float64), tall[:10].astype(np.float64)).astype(dtype)
    compare('precomputed', gram, cross, m=10, seed=5)
    compare('precomputed', gram, cross, m=20, seed=5)
    compare('precomputed', gram, cross, m=4, seed=5)
    compare('precomputed', gram, cross, gamma=0.1, m=10, seed=5)
    negative = tall.copy(); negative[0, 0] = -0.5
    compare('chi2', negative, tall_y, m=5)
    compare('additive_chi2', tall, -tall_y, m=5)
# Explicit scheduling parity, including uneven blocks and negative job counts.
for dtype in (np.float32, np.float64):
    x = r.uniform(0.1, 1, (13, 4)).astype(dtype)
    y = r.uniform(0.1, 1, (7, 4)).astype(dtype)
    compare('rbf', x, y, m=3, n_jobs=0)
    for jobs in (2, -1, -2, 50):
        for kernel in ('rbf', 'laplacian', 'linear', 'chi2'):
            compare(kernel, x, y, m=3, n_jobs=jobs)

# Independent analytic accuracy checks through and below sklearn's 1e-12
# singular-value floor. Diagonal training factors avoid hiding the small
# eigenvalues in rounding of a rotated Gram. Held-out rows excite every mode.
# The target feature Gram is Y diag(s / max(s, floor)) Y.T, independent of
# SVD sign/order/null-space choices. All outputs, not just finiteness, matter.
analytic = []
for dtype in (np.float32, np.float64):
    for scale in (1e-8, 1.0, 1e8):
        for small in (0.0, 1e-14, 0.5e-12, 1e-12, 2e-12, 1e-8):
            spectrum = np.array([scale, scale * 0.1, small, 0.0])
            x = np.diag(np.sqrt(spectrum)).astype(dtype)
            y = np.vstack([np.eye(4), r.uniform(-1, 1, (3, 4))]).astype(dtype)
            result = native('linear', x, y, None, None, None, 4, 7, 2)
            assert result.returncode == 0, result.stderr
            values = np.array(result.stdout.split(), dtype=float)
            got = values[4 + 16:-1].reshape(7, 4)
            # Include the actual input-dtype rounding of the basis dot product.
            eigenvalues = (np.diag(x) ** 2).astype(float)
            weights = np.diag(x).astype(float) ** 2 / np.maximum(eigenvalues, dtype(1e-12))
            expected = (y.astype(float) * weights) @ y.astype(float).T
            error = np.max(np.abs(got @ got.T - expected))
            bound = 128 * np.finfo(dtype).eps * max(1, np.max(np.abs(expected)))
            assert error <= bound, (dtype, scale, small, error, bound)
            analytic.append(dict(dtype=np.dtype(dtype).name, scale=scale, small=small,
                                 gram_error=float(error), bound=float(bound)))
            count += 1
report = dict(cases=count, worst_error_over_bound=worst, ill_conditioned=ill, analytic_spectra=analytic, oracle_revision=REV,
              support_wheel=__import__('sklearn').__version__)
out = ROOT / 'docs/private/nystroem/parity.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print('PASS', {k: v for k, v in report.items() if k not in ('ill_conditioned', 'analytic_spectra')}, 'ill-conditioned:', len(ill))
