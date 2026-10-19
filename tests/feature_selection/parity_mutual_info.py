"""Execute pinned sklearn MI bodies with independent SciPy neighbor queries.

Sklearn's compiled extensions are unavailable in this environment. Its estimator
bodies run unchanged; SciPy cdist supplies distances, NumPy supplies validation
and scaling, and a contingency table supplies the discrete MI oracle. Native
validation and selector contracts are exercised separately in Mojo.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from scipy.special import digamma
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
sys.path.insert(0, str(ROOT / 'tools'))
import feature_selection_checks as check


class Neighbors:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.k, self.metric = n_neighbors, metric

    def fit(self, x):
        self.x = x
        return self

    def set_params(self, n_neighbors):
        self.k = n_neighbors

    def kneighbors(self):
        distances = cdist(self.x, self.x, metric=self.metric)
        np.fill_diagonal(distances, np.inf)
        if self.k >= len(self.x):
            raise ValueError('too many neighbors')
        indices = np.argsort(distances, axis=1, kind='stable')[:, :self.k]
        return np.take_along_axis(distances, indices, axis=1), indices


class Tree:
    def __init__(self, x, metric='euclidean'):
        if not len(x):
            raise ValueError('empty tree')
        self.x, self.metric = x, metric

    def query_radius(self, x, radius, **kwargs):
        return (cdist(x, self.x, metric=self.metric) <= radius[:, None]).sum(axis=1)


def discrete_mi(x, y):
    _, a = np.unique(x, return_inverse=True)
    _, b = np.unique(y, return_inverse=True)
    table = np.zeros((a.max()+1, b.max()+1))
    np.add.at(table, (a, b), 1)
    i, j = np.nonzero(table)
    counts = table[i, j]
    return np.maximum(0, np.sum(counts / len(x) * np.log(counts * len(x) / (table.sum(1)[i] * table.sum(0)[j]))))


def scale(x, with_mean=False, copy=True):
    x = np.array(x, copy=True)
    sd = np.std(x, axis=0)
    if sd.ndim == 0:
        if sd == 0:
            sd = 1
    else:
        sd[sd < 10 * np.finfo(sd.dtype).eps] = 1
    x /= sd
    return x


def oracle():
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    source = REFERENCE / 'sklearn/feature_selection/_mutual_info.py'
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(np=np, digamma=digamma, NearestNeighbors=Neighbors, KDTree=Tree,
                     mutual_info_score=discrete_mi, scale=scale, issparse=lambda x: False,
                     check_X_y=lambda x, y, **kw: (x, y), check_array=lambda x, **kw: np.asarray(x),
                     check_classification_targets=lambda y: None,
                     check_random_state=lambda seed: np.random.RandomState(seed),
                     delayed=lambda fn: fn, Parallel=lambda **kw: list)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def main():
    namespace = oracle()
    generator = np.random.RandomState(1729)
    cases = []
    for dtype in [np.float32, np.float64]:
        for task in ['classif', 'regression']:
            for mode in ['auto', 'all', 'mixed']:
                for k in [1, 3, 7]:
                    x = generator.normal(size=(24, 4)).astype(dtype)
                    y = (np.arange(24) % 3).astype(dtype) if task == 'classif' else (x[:, 0]**2 + .1 * generator.normal(size=24)).astype(dtype)
                    if mode == 'all':
                        x[:] = generator.randint(0, 4, size=x.shape)
                        discrete = True
                    elif mode == 'mixed':
                        x[:, 1] = np.arange(24) % 3
                        x[:, 3] = 1
                        discrete = [1, 3]
                    else:
                        discrete = 'auto'
                    cases.append((task, x, y, discrete, k, False))
        # Ties/constants, singleton labels, near-zero variance, large offset,
        # negative-stride ownership, mixed input/target dtype, and one-row DD.
        for task in ['classif', 'regression']:
            x = generator.randint(0, 4, size=(20, 3)).astype(dtype)
            x[:, 0] = 1
            y = (np.arange(20) % 3).astype(dtype)
            if task == 'classif':
                y[-1] = 9
            cases.append((task, x, y, False, 3, False))
            cases.append((task, x[::-1, ::-1], y[::-1], [False, True, False], 2, True))
        x = generator.normal(size=(24, 3)).astype(dtype)
        x[:, 0] = 0
        x[:, 1] *= 1e-18
        x[:, 2] += 1e4
        cases.append(('regression', x, generator.normal(size=24).astype(np.float64), False, 4, False))
        tiny_y = (generator.normal(size=24) * 1e-18).astype(dtype)
        cases.append(('regression', generator.normal(size=(24, 2)).astype(dtype), tiny_y, False, 3, False))
    cases.append(('classif', np.array([[3., 0.]]), np.array([2.]), True, 999, False))
    header = ['import numojo as nm', 'from moxie.feature_selection import mutual_info_classif, mutual_info_regression, DiscreteFeatures']
    expected, metadata, bodies = [], [], []
    for index, (task, x, y, discrete, k, reverse) in enumerate(cases):
        dt, yt = x.dtype.name, y.dtype.name
        body = [f'@no_inline\ndef case_{index}() raises:']
        def array(name, value, dtype):
            body.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({", ".join(map(str, value.shape))}))')
            for i, v in enumerate(value.flat):
                body.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({repr(float(v))}))')
        backing = x[::-1, ::-1].copy() if reverse else x.copy()
        array('backing', backing, dt)
        array('y', y, yt)
        if reverse:
            body.append(f'    var x = backing.view_with_layout(nm.Shape({x.shape[0]}, {x.shape[1]}), nm.NDArrayStrides(-{x.shape[1]}, -1), {x.size-1})')
        else:
            body.append('    var x = backing.view()')
        if isinstance(discrete, str):
            expression = 'DiscreteFeatures(String("auto"))'
        elif isinstance(discrete, bool):
            expression = f'DiscreteFeatures({discrete})'
        else:
            dtype = 'Bool' if isinstance(discrete[0], bool) else 'Int'
            body.append(f'    var discrete: List[{dtype}] = [{", ".join(map(str, discrete))}]')
            expression = 'DiscreteFeatures(discrete^)'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reference_x = x.copy()
            scores = namespace[f'mutual_info_{task}'](reference_x, y.copy(), discrete_features=discrete, n_neighbors=k, random_state=42, copy=False)
        body.append(f'    var scores = mutual_info_{task}(x, y, discrete_features={expression}, n_neighbors={k}, random_state=42, copy=False)')
        for j in range(x.shape[1]):
            body.append(f'    print(scores.unsafe_get({j}))')
            expected.append(float(scores[j]))
            metadata.append((index, task, dt, 'score'))
        # Verify copy=False storage semantics as well as seeded jitter itself.
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                body.append(f'    print(Float64(x.unsafe_get({i} * Int(x.strides[0]) + {j} * Int(x.strides[1]))))')
                expected.append(float(reference_x[i, j] if dt == 'float64' else x[i, j]))
                metadata.append((index, task, dt, 'input'))
        bodies.append('\n'.join(body))
    output = ROOT / 'builds/mutual-info-parity'
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'parity.mojo'
    source.write_text('\n\n'.join(header + bodies + ['def main() raises:\n' + '\n'.join(f'    case_{i}()' for i in range(len(cases)))]))
    check.compile_file(ROOT, source, 'mutual-info-parity-run')
    result = check.run(ROOT / 'builds/mutual-info-parity-run', capture=True)
    (output / 'native.txt').write_text(result.stdout)
    values = np.asarray([float(line) for line in result.stdout.splitlines()])
    assert len(values) == len(expected), (len(values), len(expected))
    failures = []
    for got, want, meta in zip(values, expected, metadata):
        if not np.isclose(got, want, rtol=1e-11, atol=2e-12):
            failures.append((meta, got, want))
    (output / 'results.json').write_text(json.dumps(dict(cases=len(cases), values=len(values), failures=failures, revision=REVISION), indent=2))
    assert not failures, failures[:20]
    print(f'PASS mutual information parity: {len(cases)} cases, {len(values)} scores/input values')


if __name__ == '__main__':
    main()
