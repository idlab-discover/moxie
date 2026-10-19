"""GaussianProcessRegressor parity against the pinned local scikit-learn source.

Executes the pinned `sklearn/gaussian_process/_gpr.py` and `kernels.py` from the
local checkout (revision-checked, clean); the installed sklearn supplies only
support modules (base, validation, pairwise). SciPy's L-BFGS-B is the oracle
optimizer. Native results come from builds/gpr_driver. Env: MOXIE_WORKSPACE.
"""
from pathlib import Path
import builtins, os, subprocess, sys, types, warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get('MOXIE_WORKSPACE', ROOT.parent)).resolve()
REFERENCE = WORKSPACE / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def load_pinned(relative, name, package, extra=None):
    assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', relative], text=True).strip()
    path = REFERENCE / relative
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), package
    sys.modules[name] = module
    source = path.read_text()
    if extra:
        for old, new in extra:
            assert old in source
            source = source.replace(old, new)
    exec(builtins.compile(source, str(path), 'exec'), module.__dict__)
    return module


K = load_pinned('sklearn/gaussian_process/kernels.py', 'pinned_gp_kernels', 'sklearn.gaussian_process')
# Point the pinned GPR at the pinned kernels (not the installed wheel's).
G = load_pinned('sklearn/gaussian_process/_gpr.py', 'pinned_gp_gpr', 'sklearn.gaussian_process',
                extra=[('from sklearn.gaussian_process.kernels import RBF, Kernel\n', 'from pinned_gp_kernels import RBF, Kernel\n'),
                       ('from sklearn.gaussian_process.kernels import ConstantKernel as C\n', 'from pinned_gp_kernels import ConstantKernel as C\n')])
GPR = G.GaussianProcessRegressor
assert GPR.log_marginal_likelihood.__code__.co_filename == str(REFERENCE / 'sklearn/gaussian_process/_gpr.py')


def configure(case_id, n):
    if case_id == 0:
        return GPR()
    if case_id == 1:
        return GPR(K.ConstantKernel(1.0) * K.RBF(1.0) + K.WhiteKernel(0.1), normalize_y=True, n_restarts_optimizer=3, random_state=0)
    if case_id == 2:
        return GPR(K.ConstantKernel(1.0) * K.Matern([1.0, 2.0], nu=2.5), alpha=1e-3 * (1 + np.arange(n) % 3))
    if case_id == 3:
        return GPR(K.ExpSineSquared(1.0, 3.0, periodicity_bounds=(1.0, 10.0)) + K.WhiteKernel(1e-2), n_restarts_optimizer=2, random_state=3)
    if case_id == 4:
        return GPR(K.DotProduct(1.0) + K.WhiteKernel(0.5), normalize_y=True)
    if case_id == 5:
        return GPR(K.RBF(0.7), optimizer=None, alpha=1e-2)
    if case_id == 6:
        return GPR(K.ConstantKernel(2.0) * K.Matern(1.0, nu=0.8) + K.WhiteKernel(1e-2))
    if case_id == 7:
        return GPR(K.RationalQuadratic(1.0, 1.0), normalize_y=True)
    if case_id == 8:
        return GPR(K.ConstantKernel(1.0, 'fixed') * K.RBF(1.0, (1e-2, 1e2)), n_targets=2)
    raise ValueError(case_id)


def cases():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (20, 2)); y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) + 0.05 * rng.randn(20)
    Xt = rng.uniform(-3, 3, (6, 2))
    Y2 = np.c_[y, X[:, 0] ** 2 - X[:, 1]]
    P = np.sort(rng.uniform(0, 10, 25))[:, None]; yp = np.sin(2 * np.pi * P[:, 0] / 3.1) + 0.1 * rng.randn(25)
    Pt = np.linspace(0, 12, 5)[:, None]
    L = rng.randn(15, 3); yl = L @ np.array([1.0, -2.0, 0.5]) + 3.0 + 0.1 * rng.randn(15)
    Lt = rng.randn(4, 3)
    tiny = X[:2]; ytiny = y[:2]
    for kind in ('f64', 'f32'):
        yield f'default_{kind}', 0, kind, X, y, Xt
        yield f'restarts_normalize_{kind}', 1, kind, X, y, Xt
        yield f'matern_multi_alpha_{kind}', 2, kind, X, Y2, Xt
        yield f'periodic_{kind}', 3, kind, P, yp, Pt
    yield 'dot_product_normalize', 4, 'f64', L, yl, Lt
    yield 'no_optimizer', 5, 'f64', X, y, Xt
    yield 'matern_general_nu', 6, 'f64', X, y, Xt
    yield 'rational_quadratic', 7, 'f64', X, y, Xt
    yield 'fixed_scale_two_targets', 8, 'f64', X, Y2, Xt
    yield 'column_target', 0, 'f64', X, y[:, None], Xt
    yield 'tiny', 0, 'f64', tiny, ytiny, Xt[:2]


def matrix(A):
    A = np.asarray(A, float)
    cols = A.shape[1] if A.ndim == 2 else 0
    return f"{A.shape[0]} {cols} " + ' '.join(repr(float(v)) for v in A.ravel())


def reference(case_id, kind, X, y, Xt):
    if kind == 'f32':
        X, y, Xt = (a.astype(np.float32) for a in (X, y, Xt))
    gpr = configure(case_id, X.shape[0])
    out = {}
    out['prior_mean'], out['prior_std'] = gpr.predict(Xt, return_std=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gpr.fit(X, y)
    out['warnings'] = sum(issubclass(w.category, Warning) and 'ConvergenceWarning' in w.category.__name__ for w in caught)
    out['theta'] = gpr.kernel_.theta
    out['lml'] = gpr.log_marginal_likelihood_value_
    out['alpha_'], out['L_'] = gpr.alpha_, gpr.L_
    out['mean'], out['std'] = gpr.predict(Xt, return_std=True)
    out['cov'] = gpr.predict(Xt, return_cov=True)[1]
    out['lml_shifted'], out['lml_gradient'] = gpr.log_marginal_likelihood(gpr.kernel_.theta + 0.1, eval_gradient=True)
    out['samples'] = gpr.sample_y(Xt, 3, random_state=0)
    out['score'] = gpr.score(X, y)
    out['lml_inplace'] = gpr.log_marginal_likelihood(gpr.kernel_.theta + 0.1, clone_kernel=False)
    out['theta_inplace'] = gpr.kernel_.theta
    out['lml_cached'] = gpr.log_marginal_likelihood()
    return {k: np.asarray(v, float) for k, v in out.items()}


def mahalanobis(samples, expected):
    mean, cov = expected['mean'], expected['cov']
    if mean.ndim == 1:
        mean, cov, samples = mean[:, None], cov[:, :, None], samples[:, None, :]
    out = []
    for k in range(mean.shape[1]):
        inverse = np.linalg.pinv(cov[:, :, k], rcond=1e-10, hermitian=True)
        centered = samples[:, k, :] - mean[:, k:k + 1]
        out.append(np.einsum('is,ij,js->s', centered, inverse, centered))
    return np.array(out)


def main():
    all_cases = list(cases())
    path = ROOT / 'builds/tests/gaussian_process/regression/gpr-cases.txt'
    path.write_text('\n'.join(f'{n} {c} {k} {matrix(X)} {matrix(y)} {matrix(Xt)}' for n, c, k, X, y, Xt in all_cases) + '\n')
    output = subprocess.run([str(ROOT / 'builds/tests/gaussian_process/regression/gpr_driver'), str(path)], check=True, text=True, capture_output=True).stdout
    native = {}
    for line in output.splitlines():
        if line.startswith('R|'):
            _, name, key, shape, values = line.split('|', 4)
            if key == 'error':
                native[(name, key)] = values
                continue
            dims = tuple(int(v) for v in shape.split(',')) if shape else ()
            native[(name, key)] = np.array([float(v) for v in values.split()] if values else [], float).reshape(dims)
    passed, failed = 0, []
    for name, case_id, kind, X, y, Xt in all_cases:
        if (name, 'error') in native:
            failed.append(f'{name}: native error {native[(name, "error")]}')
            continue
        expected = reference(case_id, kind, X, y, Xt)
        f32 = kind == 'f32'
        # Optimized hyperparameters: both run SciPy-faithful L-BFGS-B to the same
        # stopping rule; the float order of kernel/LML sums differs, so the final
        # iterate agrees to optimizer resolution, not bitwise.
        opt_tol = 2e-3 if f32 else 1e-4
        # General-nu Matern uses 1e-10 forward-difference gradients in both
        # implementations; its trajectory is noisier (see docs/api.md).
        if case_id == 6:
            opt_tol = 1e-3
        for key, value in expected.items():
            actual = native[(name, key)]
            if key in ('lml', 'score', 'lml_shifted', 'lml_inplace', 'lml_cached', 'warnings'):
                actual, value = actual.reshape(()), value.reshape(())
            if key == 'samples':
                # NumPy draws z ~ N(0, I) in the same order and maps it through an
                # SVD square root A (A.T A = cov) whose singular-vector signs are
                # unspecified, so draws differ by an orthogonal transform of z.
                # The whitened norm (x - mean)^T cov^+ (x - mean) = |z|^2 is
                # invariant (full-rank cov); compare that per draw and target.
                actual, value = mahalanobis(actual, expected), mahalanobis(value, expected)
                tol = 1e-4 if not f32 else 1e-2
            elif key in ('theta', 'theta_inplace'):
                tol = opt_tol
            elif key == 'warnings':
                tol = 0
            elif key in ('prior_mean', 'prior_std'):
                tol = 1e-12
            else:
                tol = opt_tol * 10
            if actual.shape != value.shape:
                failed.append(f'{name}:{key}: shape {actual.shape} != {value.shape}')
                continue
            err = np.abs(actual - value)
            bound = tol * (1 + np.abs(value))
            if np.all(err <= bound):
                passed += 1
            else:
                failed.append(f'{name}:{key}: max err {float(np.max(err)):.3e} (tol {tol:g}) native={actual.ravel()[:4]} sklearn={value.ravel()[:4]}')
    print(f'gpr parity: {passed} checks passed, {len(failed)} failed, {len(all_cases)} cases')
    for f in failed:
        print('FAIL', f)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
