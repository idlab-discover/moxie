"""SciPy 1.18.1 search states, accepted iterates and stopping-rule comparisons."""
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy
from scipy.optimize import minimize
from scipy.optimize._dcsrch import DCSRCH, dcstep

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'builds/evaluate_nca_solver'


def native(*args):
    lines = subprocess.check_output([str(BINARY), *map(str, args)], text=True)
    result = {}
    for line in lines.splitlines():
        key, *values = line.split()
        result.setdefault(key, []).append(list(map(float, values)))
    return {key: np.array(values) for key, values in result.items()}


def file_native(mode, values):
    with tempfile.NamedTemporaryFile(mode='w', dir=ROOT / 'builds') as file:
        file.write(' '.join(format(float(x), '.17g') for x in values))
        file.flush()
        return native(mode, file.name)


def search_tests():
    # Explicit dcstep cases: higher value, reversed derivative, decreasing
    # derivative magnitude (bracketed/unbracketed), and nondecreasing magnitude.
    cases = []
    for brackt in [False, True]:
        for fp, dp in [(1., -.5), (-.1, .5), (-.1, -.5), (-.1, -2.)]:
            cases.append([0., 0., -1., 4., 2., 1., 1., fp, dp, brackt, 0., 5.])
    expected = np.array([dcstep(*case) for case in cases], dtype=float)
    assert_allclose(file_native('step', np.array(cases).ravel())['D'], expected,
                    atol=3e-15, rtol=3e-15)
    problems = [
        (lambda s: ((s - 2)**2, 2*(s - 2)), [1e-4, 1., 10.]),
        (lambda s: ((s - .2)**4, 4*(s - .2)**3), [.01, 1., 100.]),
        (lambda s: (-s, -1.), [1., 1e10]),
    ]
    count = 0
    for fun, starts in problems:
        if fun(0)[1] >= 0:
            continue
        for start in starts:
            for xtol in [.1, 1e-14, 10.]:
                search = DCSRCH(None, None, 1e-3, .9, xtol, 0., 1e10)
                step, task = start, b'START'
                values = [start, 1e-3, .9, xtol, 0., 1e10]
                expected = []
                f, g = fun(0.)
                for _ in range(100):
                    values.extend([f, g])
                    step, _, _, task = search._iterate(step, f, g, task)
                    status = 3 if task == b'FG' else 4 if task == b'CONVERGENCE' else 6
                    expected.append([step, status, search.stage, search.brackt])
                    if task != b'FG':
                        break
                    f, g = fun(step)
                assert_allclose(file_native('search', values)['S'], expected,
                                atol=3e-14, rtol=3e-14)
                count += 1
    # First-stage modified objective (decrease without sufficient decrease).
    search = DCSRCH(None, None, 1e-3, .9, .1, 0., 1e10)
    step, task = 1., b'START'
    values = [1., 1e-3, .9, .1, 0., 1e10]
    expected = []
    for f, g in [(0., -1.), (-1e-5, -.999), (-.001, -.5)]:
        values.extend([f, g])
        step, _, _, task = search._iterate(step, f, g, task)
        status = 3 if task == b'FG' else 4 if task == b'CONVERGENCE' else 6
        expected.append([step, status, search.stage, search.brackt])
    assert_allclose(file_native('search', values)['S'], expected,
                    atol=3e-14, rtol=3e-14)
    # Invalid initialization must terminate before requesting an evaluation.
    for index, value in [(0, -1.), (0, 1e11), (1, -.1), (2, -.1),
                         (3, -.1), (4, -1.), (5, -1.), (7, 0.)]:
        values = [1., 1e-3, .9, .1, 0., 1e10, 0., -1.]
        values[index] = value
        search = DCSRCH(None, None, *values[1:6])
        _, _, _, task = search._iterate(values[0], *values[6:], b'START')
        assert task.startswith(b'ERROR')
        assert file_native('search', values)['S'][0, 1] == 7
    return len(cases), count, 1, 8


def objective(kind, x):
    a, b = x
    if kind == 0:
        return (a*a+10*b*b)/2, np.array([a, 10*b])
    if kind == 1:
        return (1-a)**2+100*(b-a*a)**2, np.array([2*(a-1)-400*a*(b-a*a), 200*(b-a*a)])
    if kind == 2:
        return -a, np.array([-1., 0.])
    if kind == 3:
        return a*a/2, np.array([-a, 0.])
    if kind == 4:
        return (a**4+b**4)/4, np.array([a**3, b**3])
    return a*a/2, np.array([a if a > 1.5 else -a, 0.])


def solve_tests():
    cases = [
        # Initial gradient, ordinary convergence, and iteration-before-convergence.
        (0, [0, 0], 10, 1e-5, 15000, 20),
        (0, [3, 4], 100, 1e-9, 15000, 20),
        (0, [1, 0], 1, 1e-5, 15000, 20),
        (0, [1, 0], 2, 1e-5, 15000, 20),
        # maxfun is strict > and is checked after a complete accepted search.
        (0, [1, 0], 10, 1e-5, 1, 20),
        (0, [1, 0], 10, 1e-5, 2, 20),
        (1, [-1.2, 1], 100, 1e-8, 1, 20),
        (1, [-1.2, 1], 100, 1e-8, 15000, 20),
        (1, [-1.2, 1], 100, 0, 15000, 1),
        (4, [3, -2], 100, 1e-9, 15000, 20),
        # Warning acceptance; abnormal rollback; history-reset retry.
        (2, [1, 0], 2, 0, 15000, 20),
        (3, [1, 0], 10, 0, 15000, 20),
        (5, [2, 0], 10, 0, 15000, 20),
        (0, [3, 4], 100, .1, 15000, 20),
    ]
    statuses = set()
    for kind, x, maxiter, tol, maxfun, maxls in cases:
        calls, callbacks = [], []
        def fun(x):
            calls.append(x.copy())
            return objective(kind, x)
        ref = minimize(fun, x, method='L-BFGS-B', jac=True, tol=tol,
                       callback=lambda x: callbacks.append(x.copy()),
                       options=dict(maxiter=maxiter, maxfun=maxfun, maxls=maxls))
        got = native('solve', kind, *x, maxiter, tol, maxfun, maxls)
        result = got['R'][0]
        status = (401 if 'NORM' in ref.message else 402) if ref.success else (
            504 if 'ITERATIONS' in ref.message else 502 if 'EVALUATIONS' in ref.message else 8)
        statuses.add(status)
        context = (kind, x, maxiter, tol, maxfun, maxls, ref.message, result)
        assert tuple(result[:3]) == (ref.nit, ref.nfev, status), context
        # Two-loop inverse products and SciPy's compact factorization have
        # different rounding. These small well-conditioned fixtures allow 2e-7.
        assert_allclose(result[5:], ref.x, atol=2e-7, rtol=2e-7, err_msg=str(context))
        assert_allclose(got['E'], calls, atol=2e-7, rtol=2e-7, err_msg=str(context))
        if callbacks:
            assert_array_equal(got['C'][:, 0], np.arange(1, ref.nit+1))
            assert_allclose(got['C'][:, 1:], callbacks, atol=2e-7, rtol=2e-7)
        else:
            assert 'C' not in got
        if kind == 5:
            assert result[3] == 1, context
    assert statuses == {401, 402, 504, 502, 8}, statuses
    return len(cases)


if __name__ == '__main__':
    assert scipy.__version__ == '1.18.1', scipy.__version__
    print('PASS: SciPy', scipy.__version__, 'dcstep/search cases', search_tests(),
          'unbounded solver cases', solve_tests())
