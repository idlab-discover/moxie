"""Determinant composition regression against NumPy's independent LU oracle."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io

out = ROOT / 'builds/likelihood'
out.mkdir(exist_ok=True)
rng = np.random.default_rng(631)
cases = 0
for dtype in (np.float32, np.float64):
    for n in range(2, 9):
        for seed in range(3):
            positive = rng.normal(size=(n, n)) + 2 * n * np.eye(n)
            for mode in ('positive', 'negative', 'zero_column'):
                P = positive.astype(dtype)
                if mode == 'negative':
                    P[0] *= -1
                elif mode == 'zero_column':
                    P[:, 0] = 0
                S = np.eye(n, dtype=dtype)
                sign, logabsdet = np.linalg.slogdet(P)
                expected = (-np.sum(S * P, dtype=np.float64) + float(logabsdet) - n * np.log(2 * np.pi)) / 2 if sign > 0 else -np.inf
                path = out / f'{dtype.__name__}-{n}-{seed}-{mode}.txt'
                got = io.run(ROOT / 'builds/audit_driver', path,
                             f'{32 if dtype == np.float32 else 64} 9 {io.matrix(S)} {io.matrix(P)}')['likelihood']
                # Float32 NumPy rounds logdet to Float32; the native scalar is Float64.
                np.testing.assert_allclose(got, expected, rtol=2e-7 if dtype == np.float32 else 2e-12,
                                           atol=2e-7 if dtype == np.float32 else 2e-12)
                cases += 1
print(f'PASS: {cases} nonsymmetric likelihood cases (positive/negative/zero determinant, Float32/Float64)')

# Rank-2 NumPy broadcasting, including expansion of a 1x1 precision.
broadcast_cases = 0
for dtype in (np.float32, np.float64):
    for p in (1, 2, 5):
        P = (np.eye(p) * 2 + np.ones((p, p)) / 8).astype(dtype)
        shapes = [(1, p), (p, 1), (1, 1), (p, p)]
        if p == 1:
            shapes += [(2, 3), (0, 3), (3, 0)]
        else:
            shapes += [(p + 1, p), (p, p + 1), (0, p)]
        for rows, columns in shapes:
            S = (np.arange(rows * columns).reshape(rows, columns) / 8).astype(dtype)
            got = io.run(ROOT / 'builds/audit_driver', out / f'broadcast-{dtype.__name__}-{p}-{rows}-{columns}.txt',
                         f'{32 if dtype == np.float32 else 64} 9 {io.matrix(S)} {io.matrix(P)}')['likelihood']
            try:
                # Same expression as the pinned sklearn log_likelihood body.
                expected = (-float(np.sum(S * P)) + float(np.linalg.slogdet(P)[1])
                            - p * np.log(2 * np.pi)) / 2
            except ValueError:
                assert isinstance(got, Exception) and 'ValueError' in str(got), got
            else:
                np.testing.assert_allclose(got, expected, rtol=2e-7 if dtype == np.float32 else 2e-12,
                                           atol=2e-7 if dtype == np.float32 else 2e-12)
            broadcast_cases += 1
print(f'PASS: {broadcast_cases} likelihood broadcasting and incompatible-shape cases')
