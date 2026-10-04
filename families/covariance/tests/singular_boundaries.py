"""Isolate pseudoinverse, determinant, and alpha-zero inverse errors.

Exact binary fixtures gate singularity/sign; well-conditioned fixtures gate
numeric agreement. Ill-conditioned fixtures retain forward and backward errors
and cross-evaluated determinants as observations, without a widened tolerance
or treating finite/-inf disagreement as a pass at sklearn score parity.
"""
from pathlib import Path
import json
import sys
import warnings
from fractions import Fraction
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.linalg import pinvh, inv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io


def exact_determinant_sign(matrix):
    """Exact rational elimination on the stored floats (validation only)."""
    work = [[Fraction(float(value)) for value in row] for row in matrix]
    sign = 1
    for k in range(len(work)):
        pivot = next((i for i in range(k, len(work)) if work[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            work[k], work[pivot] = work[pivot], work[k]
            sign = -sign
        if work[k][k] < 0:
            sign = -sign
        for i in range(k + 1, len(work)):
            ratio = work[i][k] / work[k][k]
            for j in range(k + 1, len(work)):
                work[i][j] -= ratio * work[k][j]
    return sign


def main():
    out = ROOT / 'builds/singular-boundaries'
    out.mkdir(exist_ok=True)
    records = []
    for dtype in (np.float32, np.float64):
        eps = np.finfo(dtype).eps
        fixtures = {'identity': np.eye(3), 'zero': np.zeros((3, 3)),
                    'rank_one': np.ones((3, 3)), 'zero_pivot': [[0, 1], [1, 0]],
                    'negative': [[1, 0], [0, -2]], 'positive_indefinite': np.diag([-1, -2, 3]),
                    'diagonal_rank': np.diag([1, 0, 2]),
                    'cutoff_below': np.diag([1, eps]), 'cutoff_above': np.diag([1, 4 * eps])}
        # The audit's alpha-zero SPD covariance. SciPy selects Cholesky here;
        # forcing LDL.T used to differ by 1.8% despite a tiny backward error.
        if dtype == np.float64:
            X = np.random.default_rng(310).normal(size=(30, 3))
            X[:, 1] += .8 * X[:, 0]
            X[:, 1] = X[:, 0] + 1e-7 * X[:, 1]
            fixtures['audit-spd'] = np.cov(X.T, bias=True)
        # Exactly representable pivots on both sides of zero and at singularity.
        for direction in (-1, 0, 1):
            for steps in (1, 4, 32):
                fixtures[f'near-{direction}-{steps}'] = [[1, 1], [1, 1 + direction * steps * eps]]
        rng = np.random.default_rng(88)
        for n in (3, 5, 8):
            Z = rng.normal(size=(n + 2, n))
            fixtures[f'spd-{n}'] = Z.T @ Z + np.eye(n)
            Z[:, -1] = Z[:, 0] + Z[:, 1]
            fixtures[f'gram-rank-{n}'] = Z.T @ Z
            q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            fixtures[f'conditioned-{n}'] = (q * np.geomspace(1, eps, n)) @ q.T
        for name, raw in fixtures.items():
            A = np.asarray(raw, dtype=dtype)
            case = f'{dtype.__name__}-{name}'
            def native(matrix, tag):
                return io.run(ROOT / 'builds/singular_driver', out / f'{case}-{tag}.txt',
                              f'{32 if dtype == np.float32 else 64} {io.matrix(matrix)}')
            got = native(A, 'input')
            P = pinvh(A)
            N = got['pinvh'].reshape(A.shape)
            crossed = native(P, 'reference-precision')
            def logdet(matrix):
                sign, logabs = np.linalg.slogdet(matrix)
                return float(logabs) if sign > 0 else -np.inf
            def residual(matrix, inverse):
                # Scale by operand norms: forward error is not meaningful alone.
                a, b = matrix.astype(np.float64), inverse.astype(np.float64)
                return float(np.linalg.norm(a @ b - np.eye(len(a)), np.inf) /
                             max(1., np.linalg.norm(a, np.inf) * np.linalg.norm(b, np.inf)))
            record = dict(case=case, condition=float(np.linalg.cond(A.astype(np.float64))),
                          reference_precision_symmetric=bool(np.array_equal(P, P.T)),
                          exact_signs=dict(input=exact_determinant_sign(A),
                                           native_precision=exact_determinant_sign(N),
                                           reference_precision=exact_determinant_sign(P)),
                          pinvh_relative=float(np.linalg.norm(N-P) / max(np.linalg.norm(P), np.finfo(float).tiny)),
                          determinant=dict(input_native=float(got['logdet'][0]), input_reference=logdet(A),
                                           native_precision_native=float(got['pinvh_logdet'][0]),
                                           native_precision_reference=logdet(N.astype(dtype)),
                                           reference_precision_native=float(crossed['logdet'][0]),
                                           reference_precision_reference=logdet(P)))
            try:
                reference_inverse = inv(A.astype(np.float64))  # native alpha=0 workspace is Float64
            except np.linalg.LinAlgError:
                reference_inverse = None
            actual_inverse = got.get('inverse')
            record['inverse_native_failed'] = isinstance(actual_inverse, Exception)
            record['inverse_reference_failed'] = reference_inverse is None
            if not isinstance(actual_inverse, Exception):
                actual_inverse = actual_inverse.reshape(A.shape)
                record['inverse_native_residual'] = residual(A, actual_inverse)
            if reference_inverse is not None:
                record['inverse_reference_residual'] = residual(A, reference_inverse)
                if not isinstance(actual_inverse, Exception):
                    record['inverse_relative'] = float(np.linalg.norm(actual_inverse-reference_inverse) / np.linalg.norm(reference_inverse))
            # Explicit exact-domain and well-conditioned contracts, not an allowlist
            # claiming that the observations outside these gates match sklearn.
            if name == 'audit-spd':
                assert reference_inverse is not None and not isinstance(actual_inverse, Exception)
                # Existing Float64 public audit tolerance, not cond(A)*eps.
                np.testing.assert_allclose(actual_inverse, reference_inverse, rtol=2e-8, atol=2e-8)
            elif not name.startswith(('gram-', 'conditioned-', 'near-')):
                np.testing.assert_allclose(N, P, rtol=5e-6 if dtype == np.float32 else 2e-12, atol=5e-7 if dtype == np.float32 else 2e-14)
                np.testing.assert_allclose(got['logdet'], logdet(A), rtol=3e-7, atol=3e-7)
                assert isinstance(got['inverse'], Exception) == (reference_inverse is None), case
                if reference_inverse is not None:
                    np.testing.assert_allclose(actual_inverse, reference_inverse, rtol=2e-12, atol=2e-14)
            if name.startswith('near-'):
                # det([[1,1],[1,1+d]]) is exactly d on these binary fixtures.
                delta = float(A[1, 1]) - 1.
                expected = np.log(delta) if delta > 0 else -np.inf
                np.testing.assert_allclose(got['logdet'], expected, rtol=3e-7, atol=3e-7)
            records.append(record)
    (out / 'results.json').write_text(json.dumps(dict(cases=records, numpy=np.__version__), indent=2))
    mismatches = sum(r['determinant']['reference_precision_native'] != r['determinant']['reference_precision_reference']
                     and np.isfinite(r['determinant']['reference_precision_native']) != np.isfinite(r['determinant']['reference_precision_reference']) for r in records)
    print(f'PASS: {len(records)} singular/inverse probes; exact-domain gates pass; {mismatches} observed identical-precision determinant finiteness differences')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
