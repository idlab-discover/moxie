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
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--', 'sklearn/random_projection.py', 'sklearn/utils/_random.pyx'], check=True)
    spec = importlib.util.spec_from_file_location('sklearn._moxie_reference_projection', checkout / 'sklearn/random_projection.py')
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
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
            # Inversion/BTPE, complemented probabilities, permutation/tracking/
            # reservoir selection, all-zero rows, and the density=1 special case.
            for rows, cols, density in [(1, 1, 1), (5, 9, .3), (3, 1, .1), (7, 13, 1), (4, 1000, .001), (4, 1000, .01), (4, 1000, .2), (4, 1000, .7), (4, 1000, .999), (4, 1000, .99999)]:
                matrix = reference._sparse_random_matrix(rows, cols, density, seed).astype(dtype)
                emit('sparse', bits, rows, cols, seed, density, flat(matrix), flag(sparse.issparse(matrix)), flat(matrix.indices) if sparse.issparse(matrix) else '-', flat(matrix.indptr) if sparse.issparse(matrix) else '-')
        rng = np.random.RandomState(91)
        for family, cls in [('gaussian', reference.GaussianRandomProjection), ('sparse', reference.SparseRandomProjection)]:
            for rows, cols, components in [(4, 7, 3), (2, 3, 7), (3, 3, 3), (1, 1, 1), (2, 40, 'auto')]:
                for density in (['auto', 1.0, .1] if family == 'sparse' else ['auto']):
                    for sparse_input in [False, True]:
                        for cached in [False, True]:
                            dense_output = cached  # Exercise both forms independently of input.
                            X = rng.normal(size=(rows, cols)).astype(dtype)
                            X[np.abs(X) < .7] = 0
                            input_data = sparse.csr_array(X) if sparse_input else X
                            kwargs = dict(n_components=components, eps=.5, compute_inverse_components=cached, random_state=42)
                            if family == 'sparse':
                                kwargs.update(density=density, dense_output=dense_output)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                model = cls(**kwargs).fit(input_data)
                            transformed = model.transform(input_data)
                            inverse = model.inverse_transform(transformed)
                            emit('model', bits, family, rows, cols, components, .5, density, flag(dense_output), flag(cached), 42, flag(sparse_input), flat(X), model.n_components_, flat(model.components_), flat(transformed), flat(inverse), flag(sparse.issparse(transformed)), flat(model.inverse_components_) if cached else '-')
        matrices = [np.zeros((3, 5)), np.ones((5, 3)), np.array([[1, 2, 3], [2, 4, 6]]), np.eye(4), rng.normal(size=(6, 3)), rng.normal(size=(3, 6)), np.diag([1, 1e-4, 1e-8]), np.diag([1, 1e-15, 1e-17])]
        if dtype == np.float64:
            matrices += [np.eye(3) * 1e-200, np.eye(3) * 1e200]
        for matrix in matrices:
            matrix = matrix.astype(dtype)
            expected = linalg.pinv(matrix)
            emit('inverse', bits, *matrix.shape, flat(matrix), flat(expected), 2e-5 if bits == 32 else 3e-10)

    # Mixed advancing streams retain cached Gaussian draws across sparse fits.
    for dtype in [np.float32, np.float64]:
        for density in [.001, .1, .5, .999, 1.0]:
            stream = np.random.RandomState(42)
            X = np.zeros((2, 101), dtype=dtype)
            g = reference.GaussianRandomProjection(3, random_state=stream).fit(X)
            s = reference.SparseRandomProjection(3, density=density, random_state=stream).fit(X)
            g2 = reference.GaussianRandomProjection(3, random_state=stream).fit(X)
            emit('stream', 32 if dtype == np.float32 else 64, density,
                 flat(g.components_), flat(s.components_), flat(g2.components_),
                 int(stream.randint(0, 2**32, dtype=np.uint32)))

    directory = ROOT / 'builds/random_projection/parity/root'
    directory.mkdir(parents=True, exist_ok=True)
    fixture_path = directory / 'reference.txt'
    fixture_path.write_text('\n'.join(fixtures) + '\n')
    executable = directory / 'parity-runner'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', 'builds/deps', 'tests/random_projection/parity_runner.mojo', '-o', str(executable)], cwd=ROOT, check=True)
    subprocess.run([str(executable), str(fixture_path)], cwd=ROOT, check=True)
    print(f'Oracle sklearn source: {REVISION}; support sklearn={sklearn.__version__}, numpy={np.__version__}, scipy={scipy.__version__}')


if __name__ == '__main__':
    main()
