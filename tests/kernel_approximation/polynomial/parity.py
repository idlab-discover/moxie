"""Differential comparison with the pinned sklearn PolynomialCountSketch bodies.

Hashes and the next RNG draw must match exactly. Output tolerance per row:
the map is a real IFFT of a product of `degree` FFTs of count sketches, so
its normwise rounding error is O(degree * eps * log2(M)) times the product
of sketch norms, each bounded by ||x'||_1 (x' = [sqrt(gamma) x, sqrt(coef0)]):
|native - sklearn| <= 16 * degree * eps64 * log2(M) * max(1, ||x'||_1)^degree.
"""
import json, math, subprocess, tempfile, warnings
from pathlib import Path
import numpy as np
from oracle import load, ROOT, REV
Sketch = load()
count = 0
worst = 0.0


def native(x, gamma, degree, coef0, m, seed):
    bits = '32' if x.dtype == np.float32 else '64'
    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir=ROOT / 'builds', delete=False) as f:
        f.write(f'{bits} {gamma!r} {degree} {coef0!r} {m} {seed} {x.shape[0]} {x.shape[1]}\n')
        f.write(' '.join(repr(float(v)) for v in x.ravel()))
    result = subprocess.run([ROOT / 'builds/tests/kernel_approximation/polynomial/evaluate', f.name], capture_output=True, text=True)
    Path(f.name).unlink()
    return result


def compare(x, gamma=1.0, degree=2, coef0=0.0, m=16, seed=42):
    global count, worst
    rng = np.random.RandomState(seed)
    reference = Sketch(gamma=gamma, degree=degree, coef0=coef0, n_components=m, random_state=rng)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        reference.fit(x)
        want = reference.transform(x)
    result = native(x, gamma, degree, coef0, m, seed)
    label = (x.dtype, x.shape, gamma, degree, coef0, m, seed)
    if coef0 < 0:
        assert np.isnan(want).all(), label  # sklearn: sqrt(coef0) -> NaN everywhere
        assert result.returncode != 0 and 'coef0' in result.stderr + result.stdout, label
        count += 1
        return
    assert result.returncode == 0, (label, result.stderr)
    assert want.dtype == np.float64
    values = result.stdout.split()
    h = reference.indexHash_.size
    assert np.array_equal(np.array(values[:h], dtype=np.int64), reference.indexHash_.ravel()), label
    assert np.array_equal(np.array(values[h:2 * h], dtype=np.int64), reference.bitHash_.ravel()), label
    got = np.array(values[2 * h:2 * h + want.size], dtype=float).reshape(want.shape)
    assert float(values[-1]) == rng.random_sample(), (label, 'draw consumption')
    augmented = np.sqrt(gamma) * x.astype(np.float64)
    if coef0 != 0:
        augmented = np.hstack([augmented, np.full((x.shape[0], 1), np.sqrt(coef0))])
    size = m if m & (m - 1) == 0 else 1 << (2 * m - 2).bit_length()
    eps = np.finfo(np.float64).eps
    for r in range(x.shape[0]):
        bound = 16 * degree * eps * max(1, math.log2(size)) * max(1.0, np.abs(augmented[r]).sum()) ** degree
        error = float(np.max(np.abs(got[r] - want[r])))
        assert error <= bound, (label, r, error, bound)
        worst = max(worst, error / bound)
    count += 1


data = np.random.RandomState(0)
for dtype in (np.float32, np.float64):
    shapes = {'tiny': data.uniform(-1, 1, (1, 1)), 'tall': data.uniform(-1, 1, (40, 3)),
              'wide': data.uniform(0, 1, (3, 30)), 'zeros': np.zeros((2, 4))}
    for name, values in shapes.items():
        x = values.astype(dtype)
        for degree in (1, 2, 3, 5):
            for m in (1, 2, 7, 16, 100, 128):
                for coef0 in (0.0, 2.5):
                    compare(x, gamma=0.7, degree=degree, coef0=coef0, m=m, seed=degree * 7 + m)
    x = shapes['tall'].astype(dtype)
    for seed in (0, 1, 4294967295):
        compare(x, seed=seed)
    compare(x, gamma=0.0, coef0=1.0)
    compare(x, coef0=-1.0)
# Kernel approximation: the full grid of sklearn's test_polynomial_count_sketch.
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50)); Y = rng.random_sample(size=(300, 50))
X /= X.sum(axis=1)[:, np.newaxis]; Y /= Y.sum(axis=1)[:, np.newaxis]
kernel_checks = []
for gamma in (0.1, 1, 2.5):
    for degree, m in ((1, 500), (2, 500), (3, 5000)):
        for coef0 in (0, 2.5):
            result = native(np.vstack([X, Y]), float(gamma), degree, float(coef0), m, 42)
            assert result.returncode == 0, result.stderr
            values = result.stdout.split(); h = degree * (50 + (coef0 != 0))
            z = np.array(values[2 * h:2 * h + 600 * m], dtype=float).reshape(600, m)
            exact = (gamma * X @ Y.T + coef0) ** degree
            error = exact - z[:300] @ z[300:].T
            stats = dict(gamma=gamma, degree=degree, coef0=coef0, n_components=m, mean_error=float(np.mean(error)),
                         max_abs_error=float(np.max(np.abs(error))), mean_abs_error=float(np.mean(np.abs(error))))
            assert abs(stats['mean_error']) <= 0.05 and stats['max_abs_error'] <= 0.1 and stats['mean_abs_error'] <= 0.05, stats
            kernel_checks.append(stats)
report = dict(cases=count, worst_error_over_bound=worst, kernel_checks=kernel_checks, oracle_revision=REV)
out = ROOT / 'docs/private/polynomial/parity.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print('PASS', report)
