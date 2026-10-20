"""Execute pinned sklearn sample with actual NumPy/SciPy sampling operations."""
import ast
import itertools
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', SK, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
path = SK / 'sklearn/neighbors/_kde.py'
assert not subprocess.check_output(['git', '-C', SK, 'diff', 'HEAD', '--', path], text=True)
cls = next(n for n in ast.parse(path.read_text()).body
           if isinstance(n, ast.ClassDef) and n.name == 'KernelDensity')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
namespace = dict(
    np=np, gammainc=gammainc, check_is_fitted=lambda obj: None,
    check_random_state=lambda seed: seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed),
    row_norms=lambda x, squared: np.einsum('ij,ij->i', x, x),
)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
maximum = 0.0
count = 0
for dtype, dim, seed, weighted, bandwidth in itertools.product(
    ['f32', 'f64'], [1, 2, 3, 10, 63, 1000], [0, 42, 723619],
    [False, True], [0.01, 0.75, 5.0],
):
    x = np.array([[i * 10 + j % 3 for j in range(dim)] for i in range(4)],
                 dtype=np.float32 if dtype == 'f32' else np.float64)
    weights = np.array([0, 1, 0, 3.0]) if weighted else None
    model = SimpleNamespace(kernel='tophat', bandwidth_=bandwidth,
                            tree_=SimpleNamespace(data=x, sample_weight=weights))
    rng = np.random.RandomState(seed)
    expected = namespace['sample'](model, 7, rng)
    following = namespace['sample'](model, 3, rng)
    next_uniform = rng.uniform()
    raw = subprocess.check_output([
        ROOT / 'builds/test_kde_sampling', dtype, str(dim), str(seed),
        'weighted' if weighted else 'uniform', str(bandwidth),
    ], text=True)
    rows = [line.split() for line in raw.splitlines()]
    samples = np.array([float(row[1]) for row in rows if row[0] == 'S']).reshape(7, dim)
    subsequent = np.array([float(row[1]) for row in rows if row[0] == 'N']).reshape(3, dim)
    # Native pow and MT19937 Gaussian math have bounded rounding differences.
    # Scale by bandwidth so translated centers cannot hide radial errors.
    for actual, reference in [(samples, expected), (subsequent, following)]:
        error = float(np.max(np.abs(actual - reference)))
        maximum = max(maximum, error)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-9 * bandwidth + 2e-14)
    np.testing.assert_equal([float(row[1]) for row in rows if row[0] == 'U'], [next_uniform])
    centers = x[[1, 3]] if weighted else x
    nearest = np.linalg.norm(samples[:, None, :] - centers, axis=2).min(axis=1)
    assert np.all(nearest <= bandwidth * (1 + 2e-9))
    count += 1
print(f'PASS {count} tophat sampling configurations: dtypes, dimensions 1–1000, seeds, '
      f'zero/nonuniform weights, bandwidths, continued RNG state; maximum absolute error {maximum:.3g}')
