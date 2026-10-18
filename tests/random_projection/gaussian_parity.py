"""Generate fixtures from pinned sklearn source; verify them in pure Mojo.

Requires NumPy, SciPy and scikit-learn as test-only dependencies.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import scipy
from scipy import linalg, sparse
import sklearn

ROOT = Path(__file__).resolve().parents[2]
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def flat(value):
    array = value.toarray() if sparse.issparse(value) else np.asarray(value)
    return ','.join(repr(float(x)) for x in array.ravel()) or '-'


def flag(value):
    return 'true' if value else 'false'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sklearn-root', type=Path, default=ROOT.parent / 'scikit-learn')
    args = parser.parse_args()
    checkout = args.sklearn_root.resolve()
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Expected {REVISION}, got {revision}')
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--', 'sklearn/random_projection.py', 'sklearn/utils/_random.pyx', 'sklearn/tests/test_random_projection.py', 'doc/modules/random_projection.rst', 'sklearn/base.py'], check=True)
    spec = importlib.util.spec_from_file_location('sklearn._moxie_reference_projection', checkout / 'sklearn/random_projection.py')
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    assert Path(reference.GaussianRandomProjection.fit.__wrapped__.__code__.co_filename).resolve() == checkout / "sklearn/random_projection.py"
    fixtures = []
    def emit(*values):
        fixtures.append('|'.join(map(str, values)))

    for samples in [1, 2, 10, 100, 1000, 1e6]:
        for eps in [.01, .1, .2, .5, .9, .999]:
            emit('jl', samples, eps, int(reference.johnson_lindenstrauss_min_dim(samples, eps=eps)))
    emit('jl', 100, 1e-5, 368416070986)

    for dtype in [np.float32, np.float64]:
        bits = 32 if dtype == np.float32 else 64
        for seed in [0, 1, 42, 123456789, 2**32-1]:
            # Includes multiple MT19937 twists and an odd Gaussian pair count.
            for rows, cols in [(1, 1), (3, 7), (7, 3), (2, 1000), (1000, 1)]:
                matrix = reference._gaussian_random_matrix(rows, cols, seed).astype(dtype)
                emit('gaussian', bits, rows, cols, seed, '-', flat(matrix), 'false', '-', '-')
        rng = np.random.RandomState(91)
        for rows, cols, components in [(4, 7, 3), (2, 3, 7), (3, 3, 3), (1, 1, 1), (2, 40, 'auto')]:
            for cached in [False, True]:
                X = rng.normal(size=(rows, cols)).astype(dtype)
                X[np.abs(X) < .7] = 0
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = reference.GaussianRandomProjection(n_components=components, eps=.5, compute_inverse_components=cached, random_state=42).fit(X)
                transformed = model.transform(X)
                inverse = model.inverse_transform(transformed)
                emit('model', bits, 'gaussian', rows, cols, components, .5, 'auto', 'true', flag(cached), 42, 'false', flat(X), model.n_components_, flat(model.components_), flat(transformed), flat(inverse), flag(sparse.issparse(transformed)), flat(model.inverse_components_) if cached else '-')
        # Borrowed-state branch: odd first fit leaves a cached Gaussian.
        stream = np.random.RandomState(42)
        X = np.arange(28, dtype=dtype).reshape(4, 7) / 10
        model = reference.GaussianRandomProjection(3, random_state=stream, compute_inverse_components=True)
        model.fit(X)
        model.fit(X)
        transformed = model.transform(X)
        emit('model', bits, 'gaussian', 4, 7, 3, .1, 'auto', 'true', 'true', 42, 'false', flat(X), 3, flat(model.components_), flat(transformed), flat(model.inverse_transform(transformed)), 'false', flat(model.inverse_components_), 21)
        matrices = [np.zeros((3, 5)), np.ones((5, 3)), np.array([[1, 2, 3], [2, 4, 6]]), np.eye(4), rng.normal(size=(6, 3)), rng.normal(size=(3, 6)), np.diag([1, 1e-4, 1e-8]), np.diag([1, 1e-15, 1e-17])]
        if dtype == np.float64:
            matrices += [np.eye(3) * 1e-200, np.eye(3) * 1e200]
        for matrix in matrices:
            matrix = matrix.astype(dtype)
            expected = linalg.pinv(matrix)
            emit('inverse', bits, *matrix.shape, flat(matrix), flat(expected), 2e-5 if bits == 32 else 3e-10)

    directory = ROOT / 'builds/random_projection/parity/gaussian'
    directory.mkdir(parents=True, exist_ok=True)
    fixture_path = directory / 'reference.txt'
    fixture_path.write_text('\n'.join(fixtures) + '\n')
    executable = directory / 'parity-runner'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', 'builds/deps', 'tests/random_projection/gaussian_parity_runner.mojo', '-o', str(executable)], cwd=ROOT, check=True)
    subprocess.run([str(executable), str(fixture_path)], cwd=ROOT, check=True)
    print(f'Oracle sklearn source: {REVISION}; support sklearn={sklearn.__version__}, numpy={np.__version__}, scipy={scipy.__version__}')


if __name__ == '__main__':
    main()
