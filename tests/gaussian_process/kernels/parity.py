"""Differential kernel parity against the pinned local scikit-learn source.

The pinned `sklearn/gaussian_process/kernels.py` is executed from the local
checkout (revision-checked, clean) inside the installed sklearn package context;
the installed wheel supplies only support modules (pairwise metrics, base).
The native results come from builds/kernel_driver. Python is tooling only.
Environment: MOXIE_WORKSPACE, GP_PARITY_VERBOSE=1.
"""
from pathlib import Path
import builtins
import os
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get('MOXIE_WORKSPACE', ROOT.parent)).resolve()
REFERENCE = WORKSPACE / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
VERBOSE = os.environ.get('GP_PARITY_VERBOSE') == '1'


def load_pinned(relative, module_name, package):
    """Execute a pinned source file as `module_name` and return the module."""
    assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', relative], text=True).strip()
    path = REFERENCE / relative
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = package
    exec(builtins.compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


K = load_pinned('sklearn/gaussian_process/kernels.py', 'pinned_gp_kernels', 'sklearn.gaussian_process')
assert K.RBF.__call__.__code__.co_filename == str(REFERENCE / 'sklearn/gaussian_process/kernels.py')

INF = np.inf
ANISO = [0.5, 2.0, 1.2]


def pairwise(metric, gamma=0.4, **kwargs):
    return K.PairwiseKernel(gamma=gamma, metric=metric, pairwise_kernels_kwargs=kwargs or None)


# Must match catalog() in tests/kernel_driver.mojo entry for entry.
KERNELS = [
    K.ConstantKernel(2.5),
    K.ConstantKernel(2.5, 'fixed'),
    K.WhiteKernel(0.3),
    K.RBF(1.3),
    K.RBF(ANISO),
    K.RBF(1.3, 'fixed'),
    K.Matern(0.9, nu=0.5),
    K.Matern(0.9, nu=1.5),
    K.Matern(0.9, nu=2.5),
    K.Matern(0.9, nu=INF),
    K.Matern(0.9, nu=0.3),
    K.Matern(1.4, nu=3.7),
    K.Matern(ANISO, nu=0.5),
    K.Matern(ANISO, nu=1.5),
    K.Matern(ANISO, nu=2.5),
    K.Matern(ANISO, nu=INF),
    K.Matern(ANISO, nu=1.1),
    K.RationalQuadratic(0.8, alpha=2.5),
    K.RationalQuadratic(0.8, alpha=2.5, alpha_bounds='fixed'),
    K.ExpSineSquared(1.1, periodicity=2.3),
    K.ExpSineSquared(1.1, periodicity=2.3, periodicity_bounds='fixed'),
    K.DotProduct(0.7),
    pairwise('linear'),
    pairwise('rbf'),
    pairwise('laplacian'),
    pairwise('poly', 0.4, degree=2.0, coef0=0.5),
    pairwise('polynomial', 0.2),
    pairwise('sigmoid', 0.3, coef0=-0.2, degree=7.0),
    pairwise('cosine'),
    pairwise('chi2'),
    pairwise('additive_chi2'),
    K.ConstantKernel(2.0) * K.RBF(1.5) + K.WhiteKernel(0.1),
    (K.RBF(0.7) + K.DotProduct(0.3)) ** 2.0,
    K.RationalQuadratic(0.8, alpha=1.5) * K.ExpSineSquared(1.1, periodicity=2.3),
    K.Exponentiation(K.Matern(ANISO, nu=2.5), 1.7),
    K.ConstantKernel(0.5, 'fixed') + K.RBF(1.2) * 3.0,
    K.ConstantKernel(1.3) * (K.RBF(ANISO) + K.Matern(1.1, nu=0.5)) * K.ExpSineSquared(0.9, 3.0)
    + K.WhiteKernel(0.05, (1e-3, 1.0)),
    K.Sum(K.Product(K.ConstantKernel(0.7), K.Matern(1.4, nu=3.7)), K.WhiteKernel(0.2)),
    1.5 + K.RBF(0.8),
    2.0 * K.Matern(0.6, (1e-2, 10.0), nu=1.5),
    K.CompoundKernel([K.RBF(1.0), K.Matern(2.0, nu=2.5)]),
    K.CompoundKernel([K.ConstantKernel(2.0), K.WhiteKernel(0.4), K.DotProduct(1.1)]),
]


def is_finite_difference(kernel):
    """Kernels whose sklearn gradient is a 1e-10 forward difference."""
    if isinstance(kernel, K.PairwiseKernel):
        return not kernel.hyperparameter_gamma.fixed
    if isinstance(kernel, K.Matern):
        return kernel.nu not in (0.5, 1.5, 2.5, np.inf)
    for name in ('k1', 'k2', 'kernel'):
        if hasattr(kernel, name) and is_finite_difference(getattr(kernel, name)):
            return True
    return isinstance(kernel, K.CompoundKernel) and any(map(is_finite_difference, kernel.kernels))


def datasets():
    rng = np.random.RandomState(0)
    X = rng.randn(7, 3)
    X[6] = X[2]  # duplicate sample: zero distances off the diagonal
    Y = rng.randn(5, 3)
    Y[1] = X[4]  # cross zero distance
    yield 'normal', 'f64', X, Y
    yield 'positive', 'f64', np.abs(X), np.abs(Y)
    yield 'wide_scale', 'f64', X * 30.0, Y * 30.0
    yield 'tiny', 'f64', X[:1], Y[:1]
    yield 'float32', 'f32', X.astype(np.float32).astype(np.float64), Y.astype(np.float32).astype(np.float64)


def matrix_text(A):
    return f"{A.shape[0]} {A.shape[1]} " + ' '.join(repr(float(v)) for v in A.ravel())


def run_native(sets):
    path = ROOT / 'builds/tests/gaussian_process/kernels/parity-datasets.txt'
    path.write_text('\n'.join(f"{name} {kind} {matrix_text(X)} {matrix_text(Y)}" for name, kind, X, Y in sets) + '\n')
    binary = ROOT / 'builds/tests/gaussian_process/kernels/kernel_driver'
    output = subprocess.run([str(binary), str(path)], check=True, text=True, capture_output=True).stdout
    records = {}
    for line in output.splitlines():
        if not line.startswith('R|'):
            continue
        _, kernel, dataset, key, shape, values = line.split('|', 5)
        entry = (int(kernel), dataset, key)
        if shape:
            dims = tuple(int(v) for v in shape.split(','))
            data = np.array([float(v) for v in values.split()], dtype=float) if values else np.zeros(0)
            records[entry] = data.reshape(dims)
        else:
            records[entry] = values
    return records


class Tally:
    def __init__(self):
        self.passed = 0
        self.failed = []
        self.notes = []

    def check(self, label, ok, detail=''):
        if ok:
            self.passed += 1
        else:
            self.failed.append(f'{label}: {detail}')


def close(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        return False, f'shape {actual.shape} != {expected.shape}'
    if actual.size == 0:
        return True, ''
    err = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    worst = float(np.max(err - bound))
    return bool(np.all(err <= bound)), f'max excess {worst:.3e}, max abs err {float(np.max(err)):.3e}'


def reference(kernel, X, Y, float32, fd=False):
    Xr = X.astype(np.float32) if float32 else X
    Yr = Y.astype(np.float32) if float32 else Y
    out = {}
    out['repr'] = repr(kernel)
    out['stationary'] = kernel.is_stationary()
    out['vector_input'] = kernel.requires_vector_input
    out['hyperparameters'] = ''.join(f'{h.name}:{h.n_elements}:{h.fixed};' for h in kernel.hyperparameters)
    out['params'] = ''.join(f'{k};' for k in kernel.get_params().keys())
    out['theta'] = kernel.theta
    b = kernel.bounds
    out['bounds'] = b.reshape(-1, 2) if b.size else np.zeros((0, 2))
    Kx, G = kernel(Xr, eval_gradient=True)
    out['K'], out['gradient'] = np.asarray(Kx, float), np.asarray(G, float)
    if float32 and fd:
        # sklearn's 1e-10 forward difference of a float32 kernel matrix is
        # dominated by float32 rounding (errors ~1e-7/1e-10); the native engine
        # differences Float64 values. Compare against sklearn on the exactly
        # widened input instead (documented Float32 adaptation).
        out['gradient'] = np.asarray(kernel(X, eval_gradient=True)[1], float)
    out['KXY'] = np.asarray(kernel(Xr, Yr), float)
    out['diag'] = np.asarray(kernel.diag(Xr), float)
    try:
        out['K_shifted'] = np.asarray(kernel.clone_with_theta(kernel.theta + 0.1)(Xr), float)
    except AttributeError as exc:  # sklearn CompoundKernel.theta setter bug (self.k1)
        out['K_shifted'] = exc
    if float32 and not isinstance(kernel, K.CompoundKernel):
        out['K_call32'] = np.asarray(kernel(Xr), float)
    return out


def main():
    sets = list(datasets())
    native = run_native(sets)
    assert native, 'native driver produced no records'
    tally = Tally()
    for index, kernel in enumerate(KERNELS):
        fd = is_finite_difference(kernel)
        for name, kind, X, Y in sets:
            label = f'{index}:{kernel!r}:{name}'
            float32 = kind == 'f32'
            try:
                expected = reference(kernel, X, Y, float32, fd)
            except ValueError as exc:
                got = native.get((index, name, 'error'))
                tally.check(label + ':error', got is not None, f'sklearn raised {exc!r}, native did not')
                continue
            error = native.get((index, name, 'error'))
            if error is not None:
                tally.check(label, False, f'native raised {error}')
                continue
            for key in ('repr', 'hyperparameters', 'params'):
                tally.check(f'{label}:{key}', native[(index, name, key)] == expected[key],
                            f'{native[(index, name, key)]!r} != {expected[key]!r}')
            for key in ('stationary', 'vector_input'):
                tally.check(f'{label}:{key}', native[(index, name, key)] == str(bool(expected[key])))
            # Float32 inputs: sklearn keeps some intermediates in float32 (e.g. DotProduct inner).
            rtol, atol = (2e-6, 2e-6) if float32 else (1e-12, 1e-13)
            for key in ('theta', 'bounds'):
                ok, detail = close(native[(index, name, key)], np.asarray(expected[key], float), 1e-14, 0)
                tally.check(f'{label}:{key}', ok, detail)
            for key in ('K', 'KXY', 'diag', 'K_call32'):
                if key in expected:
                    ok, detail = close(native[(index, name, key)], expected[key], rtol, atol)
                    tally.check(f'{label}:{key}', ok, detail)
            if isinstance(expected['K_shifted'], Exception):
                tally.notes.append(f'{label}: sklearn clone_with_theta raises {expected["K_shifted"]!r}; native splits theta per child')
            else:
                ok, detail = close(native[(index, name, 'K_shifted')], expected['K_shifted'], rtol, atol)
                tally.check(f'{label}:K_shifted', ok, detail)
            # Forward differences with step 1e-10 amplify O(1e-16) value differences
            # by 1e10; compare those gradients at 2e-5 absolute (see docs/api.md).
            g_rtol, g_atol = ((2e-5, 2e-5) if fd else (rtol * 10, atol * 10))
            if float32:
                g_rtol, g_atol = (2e-5, 2e-5)
            ok, detail = close(native[(index, name, 'gradient')], expected['gradient'], g_rtol, g_atol)
            tally.check(f'{label}:gradient', ok, detail)
    for note in tally.notes if VERBOSE else tally.notes[:3]:
        print('NOTE', note)
    print(f'kernel parity: {tally.passed} checks passed, {len(tally.failed)} failed, '
          f'{len(KERNELS)} kernels x {len(sets)} datasets, {len(native)} native records')
    for failure in tally.failed[:60]:
        print('FAIL', failure)
    if tally.failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
