"""Native warning transport and verbose regression contracts (no Python runtime in library)."""
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io

out = ROOT / 'builds/diagnostics'
out.mkdir(exist_ok=True)
X = np.random.default_rng(44).normal(size=(40, 2)) + [1, -2]
for kind in range(8):
    data = X[:1] if kind < 4 else (np.column_stack([X[:, 0], X[:, 0]]) if kind < 6 else X)
    path = out / f'{kind}.txt'
    path.write_text(f'64 {kind} {io.matrix(data)} {io.matrix(data)} verbose')
    proc = subprocess.run([str(ROOT / 'builds/audit_driver'), str(path)], capture_output=True, text=True, check=True)
    category = 'UserWarning:' if kind < 6 else 'ConvergenceWarning:'
    assert category in proc.stderr, (kind, proc.stderr)
    if kind < 4:
        assert 'Only one sample available' in proc.stderr
        assert proc.stderr.count('UserWarning:') == (2 if kind == 2 else 1)
    elif kind < 6:
        assert 'not full rank' in proc.stderr
    else:
        # No warnings from the CV grid (whose iteration budget is zero).
        assert proc.stderr.count('ConvergenceWarning:') == 1, proc.stderr
        assert 'did not converge after 1 iteration' in proc.stderr
    assert bool(proc.stdout) == (kind == 6), (kind, proc.stdout)
    if kind == 6:
        assert '[graphical_lasso] Iteration 0' in proc.stdout
        assert 'cost' in proc.stdout and 'dual gap' in proc.stdout
    if kind == 7:
        assert '[GraphicalLassoCV] Done refinement' in proc.stderr
print('PASS: warning categories, singleton counts, graphical verbose and CV warning suppression')

# Zero iterations warns on each positive-alpha solve, but alpha=0 returns
# directly. Converged nonverbose solves do not print progress or warnings.
for iterations in (0, 100):
    path = out / f'iterations-{iterations}.txt'
    path.write_text(f'64 {io.matrix(X)} {io.matrix(X)} {io.matrix(np.cov(X.T, bias=True))} '
                    f'{io.real(10.)} {iterations} cd')
    proc = subprocess.run([str(ROOT / 'builds/audit_graphical_driver'), str(path)],
                          capture_output=True, text=True, check=True)
    assert proc.stderr.count('ConvergenceWarning:') == (3 if iterations == 0 else 0), proc.stderr
    assert '[graphical_lasso]' not in proc.stdout
print('PASS: zero-iteration warnings, alpha-zero early return and converged quiet fits')
