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


def classical_parity():
    binary = ROOT / 'builds/classical_mds_driver'
    compile_file(ROOT / 'tests/classical_mds_driver.mojo', binary)
    folder = ROOT / 'builds/classical-mds-parity'
    folder.mkdir(exist_ok=True)
    results = []
    for dtype in (np.float32, np.float64):
        for seed in range(4):
            rng = np.random.RandomState(seed)
            x = rng.normal(size=(13, 4)).astype(dtype)
            for metric in ('euclidean', 'manhattan', 'cosine', 'precomputed', 'minkowski'):
                data = pairwise_distances(x).astype(dtype) if metric == 'precomputed' else x
                write_matrix(folder / 'x.txt', data)
                options = {'p': 1.5} if metric == 'minkowski' else None
                model = classical.ClassicalMDS(2, metric=metric, metric_params=options)
                wanted = model.fit_transform(data)
                output = subprocess.check_output([str(binary), np.dtype(dtype).name,
                    str(folder / 'x.txt'), '2', metric, *(['1.5'] if options else [])], text=True, env=environment(ROOT))
                coordinates, eigenvalues, distances = [], [], []
                for line in output.splitlines():
                    name, value = line.split()
                    {'coordinate': coordinates, 'eigenvalue': eigenvalues, 'distance': distances}[name].append(float(value))
                got = np.asarray(coordinates).reshape(wanted.shape)
                distance_error = float(np.max(np.abs(np.asarray(distances).reshape(data.shape[0], -1) - model.dissimilarity_matrix_)))
                tol = 5e-6 if dtype == np.float32 else (2e-8 if metric == 'minkowski' else 2e-11)
                np.testing.assert_allclose(np.asarray(distances).reshape(model.dissimilarity_matrix_.shape),
                    model.dissimilarity_matrix_, rtol=tol, atol=tol)
                np.testing.assert_allclose(eigenvalues, model.eigenvalues_, rtol=tol, atol=tol)
                np.testing.assert_allclose(got, wanted, rtol=tol, atol=tol)
                results.append(dict(dtype=np.dtype(dtype).name, seed=seed, metric=metric,
                    max_coordinate_error=float(np.max(np.abs(got - wanted))), max_distance_error=distance_error))
    (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: {len(results)} ClassicalMDS comparisons to {REVISION}')


if __name__ == '__main__':
    main()
