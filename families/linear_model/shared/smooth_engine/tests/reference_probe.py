"""Execute the pinned Newton inner solve to specify the missing owner contract."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings
import numpy as np
import scipy
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[4] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILE = 'sklearn/linear_model/_glm/_newton_solver.py'
assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', FILE], text=True).strip()
source = REFERENCE / FILE
parsed = ast.parse(source.read_text(), filename=str(source))
cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'NewtonCholeskySolver')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'inner_solve')
namespace = {'np': np, 'scipy': scipy, 'warnings': warnings, 'ConvergenceWarning': RuntimeWarning}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
inner = namespace['inner_solve']
assert inner.__code__.co_filename == str(source)
for dtype in (np.float32, np.float64):
    for name, h, fallback in (
        ('regular', [[2, 1], [1, 2]], False),
        ('pivot', [[0, 1], [1, 0]], False),
        ('singular', [[1, 1], [1, 1]], True),
        ('ill_conditioned', [[1, 0], [0, 1e-30]], True),
    ):
        h = np.array(h, dtype=dtype)
        g = np.array([-1, -1], dtype=dtype)
        state = SimpleNamespace(hessian_warning=False, verbose=0, iteration=1,
                                is_multinomial_no_penalty=False,
                                is_multinomial_with_intercept=False,
                                use_fallback_lbfgs_solve=False,
                                gradient=g, prepare_gradient_hessian=lambda: (g, h))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            inner(state, None, None, None)
        assert state.use_fallback_lbfgs_solve == fallback, name
        if fallback:
            assert any('singular or very ill-conditioned' in str(w.message) for w in caught)
        else:
            np.testing.assert_allclose(h @ state.coef_newton, -g, rtol=1e-6, atol=1e-6)
        print(dtype.__name__, name, 'fallback=', fallback)
output = subprocess.check_output([str(ROOT / 'builds/numerics_probe')], text=True)
print(output)
lines = output.splitlines()
assert len(lines) == 4
for line in lines:
    fields = line.split()
    assert fields[0] in ('singular', 'pivot')
    assert not np.all(np.isfinite([float(v) for v in fields[1:]])), line
print('PASS: reference diagnostics and excluded NuMojo behavior reproduced; no engine parity claim')
