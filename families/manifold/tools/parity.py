"""Compare native diagnostics to the recorded local sklearn implementation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn.manifold
from sklearn.metrics import pairwise_distances
from check import ROOT, assemble, compile_file, environment

sys.dont_write_bytecode = True
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def load_reference(relative):
    source = REFERENCE / relative
    expected = subprocess.check_output(['git', '-C', str(REFERENCE), 'show', f'{REVISION}:{relative}'])
    assert source.read_bytes() == expected, 'reference differs from pinned revision'
    spec = importlib.util.spec_from_file_location('sklearn.manifold._moxie_' + source.stem, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_reference('sklearn/manifold/_t_sne.py')
classical = load_reference('sklearn/manifold/_classical_mds.py')


def write_matrix(path, x):
    path.write_text(f'{x.shape[0]} {x.shape[1]}\n' + ' '.join(repr(float(v)) for v in x.flat))


def main():
    assemble(ROOT)
    binary = ROOT / 'builds/trustworthiness_driver'
    compile_file(ROOT / 'tests/trustworthiness_driver.mojo', binary)
    folder = ROOT / 'builds/trustworthiness-parity'
    folder.mkdir(exist_ok=True)
    results = []
    for dtype in (np.float32, np.float64):
        for seed in range(4):
            rng = np.random.RandomState(seed)
            x = rng.normal(size=(17, 4)).astype(dtype)
            y = rng.normal(size=(17, 2)).astype(dtype)
            for metric in ('euclidean', 'sqeuclidean', 'manhattan', 'cosine', 'precomputed'):
                data = pairwise_distances(x).astype(dtype) if metric == 'precomputed' else x
                for k in (1, 5, 8):
                    write_matrix(folder / 'x.txt', data)
                    write_matrix(folder / 'y.txt', y)
                    wanted = reference.trustworthiness(data, y, n_neighbors=k, metric=metric)
                    got = float(subprocess.check_output([str(binary), np.dtype(dtype).name,
                        str(folder / 'x.txt'), str(folder / 'y.txt'), str(k), metric],
                        text=True, env=environment(ROOT)))
                    # Scores depend on integer ranks; these untied inputs require
                    # identical ranks, leaving only scalar output rounding.
                    np.testing.assert_allclose(got, wanted, rtol=0, atol=1e-14)
                    results.append(dict(dtype=np.dtype(dtype).name, seed=seed, metric=metric,
                                        neighbors=k, expected=float(wanted), actual=got))
    (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} trustworthiness comparisons to {REVISION}')
    classical_parity()


def run_classical(binary, folder, data, dtype, n_components, metric, p=None, check=True):
    write_matrix(folder / 'x.txt', data)
    result = subprocess.run([str(binary), np.dtype(dtype).name, str(folder / 'x.txt'), str(n_components),
        metric, *([repr(float(p))] if p is not None else [])], text=True, env=environment(ROOT), capture_output=True)
    if not check:
        return result
    assert result.returncode == 0, result.stderr
    coordinates, eigenvalues, distances = [], [], []
    for line in result.stdout.splitlines():
        name, value = line.split()
        {'coordinate': coordinates, 'eigenvalue': eigenvalues, 'distance': distances}[name].append(float(value))
    return np.asarray(coordinates), np.asarray(eigenvalues), np.asarray(distances)


def classical_tolerance(dtype, metric, p):
    # Float32 dissimilarities round before the float64 eigensolve. The locked
    # Mojo pow root differs from SciPy by about 1e-10 relative for fractional
    # Minkowski powers; p in {1, 2, inf} uses exact sums, abs maxima and sqrt.
    if dtype == np.float32:
        return 5e-6
    if metric == 'minkowski' and p not in (None, 1.0, 2.0, np.inf):
        return 2e-8
    return 2e-11


def classical_parity():
    binary = ROOT / 'builds/classical_mds_driver'
    compile_file(ROOT / 'tests/classical_mds_driver.mojo', binary)
    folder = ROOT / 'builds/classical-mds-parity'
    folder.mkdir(exist_ok=True)
    results = []
    cases = [('euclidean', None), ('l2', None), ('sqeuclidean', None), ('manhattan', None),
             ('l1', None), ('cityblock', None), ('chebyshev', None), ('cosine', None),
             ('precomputed', None), ('minkowski', None), ('minkowski', 1.0), ('minkowski', 1.5),
             ('minkowski', 2.0), ('minkowski', 3.0), ('minkowski', np.inf)]
    for dtype in (np.float32, np.float64):
        for seed in range(4):
            rng = np.random.RandomState(seed)
            x = rng.normal(size=(13, 4)).astype(dtype)
            for metric, p in cases:
                data = pairwise_distances(x).astype(dtype) if metric == 'precomputed' else x
                options = None if p is None else {'p': p}
                model = classical.ClassicalMDS(2, metric=metric, metric_params=options)
                wanted = model.fit_transform(data)
                coordinates, eigenvalues, distances = run_classical(binary, folder, data, dtype, 2, metric, p)
                got = coordinates.reshape(wanted.shape)
                tol = classical_tolerance(dtype, metric, p)
                distance_error = float(np.max(np.abs(distances.reshape(model.dissimilarity_matrix_.shape) - model.dissimilarity_matrix_)))
                np.testing.assert_allclose(distances.reshape(model.dissimilarity_matrix_.shape),
                    model.dissimilarity_matrix_, rtol=tol, atol=tol)
                np.testing.assert_allclose(eigenvalues, model.eigenvalues_, rtol=tol, atol=tol)
                np.testing.assert_allclose(got, wanted, rtol=tol, atol=tol)
                results.append(dict(dtype=np.dtype(dtype).name, seed=seed, metric=metric, p=p,
                    max_coordinate_error=float(np.max(np.abs(got - wanted))), max_distance_error=distance_error))
    results.extend(classical_edges(binary, folder))
    (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} ClassicalMDS comparisons to {REVISION}')


def classical_edges(binary, folder):
    results = []
    for dtype in (np.float32, np.float64):
        tol = classical_tolerance(dtype, 'euclidean', None)
        x = np.random.RandomState(7).normal(size=(13, 4)).astype(dtype)
        zero_row = x.copy()
        zero_row[3] = 0
        duplicate = x.copy()
        duplicate[5] = duplicate[2]
        fixtures = [('more components', x, 3, 'euclidean'), ('zero row', zero_row, 2, 'cosine'),
                    ('duplicate rows', duplicate, 2, 'euclidean'),
                    ('constant', np.ones((6, 3), dtype=dtype), 2, 'euclidean'),
                    ('single sample', x[:1], 2, 'euclidean')]
        for name, data, k, metric in fixtures:
            model = classical.ClassicalMDS(k, metric=metric)
            wanted = model.fit_transform(data)
            coordinates, eigenvalues, _ = run_classical(binary, folder, data, dtype, k, metric)
            np.testing.assert_allclose(eigenvalues, model.eigenvalues_, rtol=tol, atol=tol)
            np.testing.assert_allclose(coordinates.reshape(wanted.shape), wanted, rtol=tol, atol=tol)
            results.append(dict(dtype=np.dtype(dtype).name, edge=name, shape=list(wanted.shape)))
        # More components than samples keeps one per sample. Eigenvalues past the
        # rank (4 centered features) are roundoff whose sign decides NaN, and
        # their eigenvectors span a degenerate null space, so only the shape,
        # the leading spectrum and the leading coordinates are comparable.
        model = classical.ClassicalMDS(20)
        with np.errstate(invalid='ignore'):
            wanted = model.fit_transform(x)
        coordinates, eigenvalues, _ = run_classical(binary, folder, x, dtype, 20, 'euclidean')
        assert wanted.shape == (13, 13) and coordinates.size == wanted.size and eigenvalues.size == 13
        np.testing.assert_allclose(eigenvalues[:4], model.eigenvalues_[:4], rtol=tol, atol=tol)
        np.testing.assert_allclose(np.abs(eigenvalues[4:]), 0, atol=1e-4 if dtype == np.float32 else 1e-12)
        np.testing.assert_allclose(coordinates.reshape(wanted.shape)[:, :4], wanted[:, :4], rtol=tol, atol=tol)
        results.append(dict(dtype=np.dtype(dtype).name, edge='components above samples', shape=[13, 13]))
        # The reference rejects asymmetric precomputed input; Moxie must also fail.
        asymmetric = pairwise_distances(x).astype(dtype)
        asymmetric[0, 1] += 1e-3
        try:
            classical.ClassicalMDS(2, metric='precomputed').fit(asymmetric)
            raise AssertionError('reference accepted an asymmetric matrix')
        except ValueError as error:
            assert 'symmetric' in str(error)
        failure = run_classical(binary, folder, asymmetric, dtype, 2, 'precomputed', check=False)
        assert failure.returncode and 'symmetric' in failure.stderr + failure.stdout, failure
        # Metrics outside the documented dense subset fail clearly.
        failure = run_classical(binary, folder, x, dtype, 2, 'braycurtis', check=False)
        assert failure.returncode and 'unsupported ClassicalMDS metric' in failure.stderr + failure.stdout, failure
        results.append(dict(dtype=np.dtype(dtype).name, edge='errors'))
    return results


if __name__ == '__main__':
    main()
