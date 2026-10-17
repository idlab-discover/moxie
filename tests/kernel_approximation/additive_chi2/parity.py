"""Differential comparison of the native map against the pinned sklearn method bodies.

Tolerance: every output entry is a short chain of correctly-rounded-ish
elementary functions (sqrt, log, cos/sin) of the same input-dtype operands,
so the native and NumPy results may differ by a few ulp of the output scale:
|got - want| <= 16 * eps(dtype) * max(1, max|want|). Feature names must match exactly.
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from oracle import load, ROOT, REV
Sampler = load()
count = 0
worst = {'float32': 0.0, 'float64': 0.0}


def native(x, steps, interval):
    bits = '32' if x.dtype == np.float32 else '64'
    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir=ROOT / 'builds', delete=False) as f:
        f.write(' '.join([bits, str(steps), 'none' if interval is None else repr(float(interval)), *map(str, x.shape)]))
        f.write('\n' + ' '.join(repr(float(v)) for v in x.ravel()))
    result = subprocess.run([ROOT / 'builds/tests/kernel_approximation/additive_chi2/evaluate', f.name], capture_output=True, text=True)
    Path(f.name).unlink()
    return result


def compare(x, steps=2, interval=None):
    global count
    x = np.asarray(x)
    reference = Sampler(sample_steps=steps, sample_interval=interval)
    try:
        want = reference.fit(x).transform(x)
        want_names = list(reference.get_feature_names_out())
    except ValueError as error:
        result = native(x, steps, interval)
        assert result.returncode != 0 and 'ValueError' in result.stderr + result.stdout, (steps, interval, error, result.stdout)
        count += 1
        return
    result = native(x, steps, interval)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    got = np.array(lines[:want.size], dtype=float).reshape(want.shape)
    names = lines[want.size:]
    assert names == want_names, (names[:4], want_names[:4])
    eps = np.finfo(x.dtype).eps
    bound = 16 * eps * max(1.0, float(np.max(np.abs(want))))
    error = float(np.max(np.abs(got - want)))
    assert error <= bound, (x.dtype, x.shape, steps, interval, error, bound)
    worst[np.dtype(x.dtype).name] = max(worst[np.dtype(x.dtype).name], error)
    count += 1
    return got


rng = np.random.RandomState(0)
for dtype in (np.float32, np.float64):
    datasets = {
        'tall': rng.uniform(0, 1, (60, 4)),
        'wide': rng.uniform(0, 3, (3, 25)),
        'tiny': np.array([[0.7]]),
        'sparse-pattern': rng.uniform(0, 1, (20, 6)) * (rng.uniform(size=(20, 6)) > 0.5),
        'wide-range': np.array([[1e-30, 1e-8, 1.0], [1e3, 1e6, 0.0]]),
        'all-zero': np.zeros((2, 3)),
    }
    for name, data in datasets.items():
        x = data.astype(dtype)
        for steps, interval in [(1, None), (2, None), (3, None), (4, 0.3), (5, 1.7), (2, 0.0), (1, 2.5)]:
            compare(x, steps, interval)
    # Reference parameter/domain errors must also raise natively.
    good = datasets['tall'][:3].astype(dtype)
    compare(good, 4, None)
    compare(good, 7, None)
    negative = good.copy(); negative[1, 2] = -1e-6
    compare(negative, 2, None)
# Kernel approximation: <phi(x), phi(y)> approximates sum 2xy/(x+y) (sklearn's test).
x = rng.uniform(0, 1, (30, 50)); y = rng.uniform(0, 1, (30, 50))
x /= x.sum(axis=1)[:, None]; y /= y.sum(axis=1)[:, None]  # histograms, as in sklearn's test
exact = (2 * x[:, None, :] * y[None, :, :] / (x[:, None, :] + y[None, :, :])).sum(axis=2)
both = compare(np.vstack([x, y]), 3, None)
approx = both[:30] @ both[30:].T
kernel_error = float(np.max(np.abs(approx - exact)))
# sklearn's own test uses assert_array_almost_equal(decimal=1): |error| < 1.5e-1.
assert kernel_error < 0.15, kernel_error
report = dict(cases=count, max_absolute_error=worst, kernel_max_absolute_error_steps3=kernel_error, oracle_revision=REV)
out = ROOT / 'docs/private/additive_chi2/parity.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n')
print('PASS', report)
