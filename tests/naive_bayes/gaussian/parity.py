"""Generate native Gaussian fixtures from the pinned sklearn checkout.

Run with builds/naive-bayes-reference-env/bin/python tests/naive_bayes/gaussian/parity.py.
Float64: 1e-7; Float32 oracle: 3e-5 (same-dtype reductions; no bitwise SIMD/libm claim).
"""
from pathlib import Path
import inspect
import json
import re
import subprocess
import warnings

import numpy as np
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[3]
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
REFERENCE = ROOT.parent / 'scikit-learn'
assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert Path(inspect.getfile(GaussianNB)).read_bytes() == (REFERENCE / 'sklearn/naive_bayes.py').read_bytes()
OUT = ROOT / 'builds/gaussian-parity'
OUT.mkdir(parents=True, exist_ok=True)


def literal(x):
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if np.isscalar(x):
        if np.isnan(x):
            return 'Float64("nan")'
        if np.isposinf(x):
            return 'Float64("inf")'
        if np.isneginf(x):
            return '-Float64("inf")'
        return repr(float(x))
    return '[' + ', '.join(literal(v) for v in x) + ']'


lines = ['''import numojo as nm
from numojo import Shape, Item
from std.math import isnan, isfinite
from std.testing import assert_equal, assert_true
from moxie.naive_bayes import GaussianNB, NumericDomain, StringDomain
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import dense_from_lists, ClassDomain

@no_inline
def close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    if isnan(expected):
        assert_true(isnan(actual))
    elif not isfinite(expected):
        assert_equal(actual, expected)
    else:
        if not isfinite(actual) or abs(actual - expected) > tolerance * (1 + abs(expected)):
            print("Mismatch", actual, expected, tolerance)
        assert_true(isfinite(actual) and abs(actual - expected) <= tolerance * (1 + abs(expected)))

@no_inline
def check_vector[dtype: DType](actual: nm.NDArray[dtype], expected: List[Float64], tolerance: Float64) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.shape[0], len(expected))
    for i in range(len(expected)):
        close(Float64(actual.item(i)), expected[i], tolerance)

@no_inline
def check_matrix[dtype: DType](actual: nm.NDArray[dtype], expected: List[List[Float64]], tolerance: Float64) raises:
    assert_equal(actual.ndim, 2)
    assert_equal(actual.shape[0], len(expected))
    assert_equal(actual.shape[1], len(expected[0]))
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            close(Float64(actual.item(i, j)), expected[i][j], tolerance)

@no_inline
def train[dtype: DType, D: ClassDomain](mut model: GaussianNB[D, dtype], X: nm.NDArray[dtype], y: D.Target,
    classes: Optional[D.Target], weights: Optional[List[Float64]], incremental: Bool) raises:
    if incremental:
        model.partial_fit(X, y, classes=classes, sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)

@no_inline
def call_predict_joint_log_proba[dtype: DType, D: ClassDomain](model: GaussianNB[D, dtype], X: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    return model.predict_joint_log_proba(X)

@no_inline
def call_predict_log_proba[dtype: DType, D: ClassDomain](model: GaussianNB[D, dtype], X: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    return model.predict_log_proba(X)

@no_inline
def call_predict_proba[dtype: DType, D: ClassDomain](model: GaussianNB[D, dtype], X: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    return model.predict_proba(X)

@no_inline
def call_predict[dtype: DType, D: ClassDomain](model: GaussianNB[D, dtype], X: nm.NDArray[dtype]) raises -> D.Target:
    return model.predict(X)

@no_inline
def call_score[dtype: DType, D: ClassDomain](model: GaussianNB[D, dtype], X: nm.NDArray[dtype], y: D.Target,
    weights: List[Float64]) raises -> Float64:
    return model.score(X, y, sample_weight=weights.copy())
''']
names = []


def case(X, y, Q, *, dtype='float64', strings=False, chunks=None, weights=None,
         priors=None, smoothing=1e-9, classes=None):
    ident = f'case_{len(names)}'
    names.append(ident)
    X, Q = np.asarray(X, dtype=dtype), np.asarray(Q, dtype=dtype)
    y = np.asarray(y)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if strings:
        convert = lambda v: np.array([f'label_{i}' for i in v])
        y, classes = convert(y), convert(classes)
    model = GaussianNB(priors=priors, var_smoothing=smoothing)
    domain = 'StringDomain' if strings else 'NumericDomain[DType.float64]'
    lines.extend([f'\n@no_inline\ndef {ident}() raises:', f'    print("{ident}")',
                  f'    var model = GaussianNB[{domain}, DType.{dtype}](var_smoothing={smoothing!r}' +
                  (f', priors=List[Float64]({literal(priors)})' if priors is not None else '') + ')'])
    def target(name, values):
        if strings:
            lines.append(f'    var {name} = StringTarget({literal(values)})')
        else:
            lines.append(f'    var {name} = nm.zeros[DType.float64](Shape({len(values)}))')
            for i, v in enumerate(values):
                lines.append(f'    {name}[Item({i})] = {literal(v)}')
    if chunks is not None:
        target('classes', classes)
    def compare(expr, expected, tol):
        expected = np.asarray(expected)
        if expected.ndim == 0:
            lines.append(f'    close(Float64({expr}), {literal(expected.item())}, {tol})')
            return
        helper = 'check_vector' if expected.ndim == 1 else 'check_matrix'
        lines.append(f'    {helper}({expr}, {literal(expected)}, {tol})')
    def predictions():
        lines.append(f'    var Q = dense_from_lists({literal(Q)}).astype[DType.{dtype}]()')
        for i, value in enumerate(model.classes_):
            if strings:
                lines.append(f'    assert_equal(model.classes_.get({i}), {literal(value)})')
            else:
                lines.append(f'    assert_equal(model.classes_.item({i}), {literal(value)})')
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            prediction = model.predict(Q)
            lines.append('    var prediction = call_predict(model, Q)')
            for i, value in enumerate(prediction):
                method = 'get' if strings else 'item'
                lines.append(f'    assert_equal(prediction.{method}({i}), {literal(value)})')
            for method in ['predict_joint_log_proba', 'predict_log_proba', 'predict_proba']:
                lines.append(f'    var {method} = call_{method}(model, Q)')
                expected = getattr(model, method)(Q)
                assert expected.dtype == np.dtype(dtype), method
                compare(method, expected, tolerance)
            target('score_y', np.resize(y, len(Q)))
            score_weights = np.arange(1, len(Q) + 1, dtype=float)
            lines.append(f'    var score_weights: List[Float64] = {literal(score_weights)}')
            compare('call_score(model, Q, score_y, score_weights)',
                    model.score(Q, np.resize(y, len(Q)), sample_weight=score_weights), tolerance)

    tolerance = '3e-5' if dtype == 'float32' else '1e-7'
    slices = chunks if chunks is not None else [slice(None)]
    for k, sl in enumerate(slices):
        lines.append(f'    var X{k} = dense_from_lists({literal(X[sl])}).astype[DType.{dtype}]()')
        target(f'y{k}', y[sl])
        kw = 'None'
        if weights is not None:
            kw = f'List[Float64]({literal(np.asarray(weights)[sl])})'
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            if chunks is None:
                model.fit(X[sl], y[sl], sample_weight=None if weights is None else np.asarray(weights)[sl])
                lines.append(f'    train(model, X{k}, y{k}, None, {kw}, False)')
            else:
                model.partial_fit(X[sl], y[sl], classes=classes if k == 0 else None,
                                  sample_weight=None if weights is None else np.asarray(weights)[sl])
                cls = 'classes.copy()' if k == 0 else 'None'
                lines.append(f'    train(model, X{k}, y{k}, {cls}, {kw}, True)')
        for attr in ['theta_', 'var_', 'class_count_', 'class_prior_', 'epsilon_']:
            assert np.asarray(getattr(model, attr)).dtype == np.dtype(dtype), attr
            compare('model.' + attr, getattr(model, attr), tolerance)
        # Scope per-update temporaries without changing the fitted model.
        lines.append('    if True:')
        start = len(lines)
        predictions()
        lines[start:] = ['    ' + line for line in lines[start:]]


rng = np.random.default_rng(672)
for dtype in ['float64', 'float32']:
    for strings in [False, True]:
        for weighted in [False, True]:
            for incremental in [False, True]:
                X = rng.normal(size=(15, 3))
                y = np.array([9, -4, 2] * 5)
                Q = rng.normal(size=(4, 3))
                for smoothing in [1e-9, 0.2]:
                    case(X, y, Q, dtype=dtype, strings=strings,
                         weights=rng.uniform(0.1, 3, 15) if weighted else None,
                         chunks=[slice(0, 3), slice(3, 8), slice(8, 15)] if incremental else None,
                         smoothing=smoothing)
# Large-offset Float32 data detects widened reductions, even with loose rtol.
for weighted in [False, True]:
    X = np.array([[1e7, 0], [1e7 + 1, 1], [1e7 + 2, 2],
                  [1e7 + 4, 3], [1e7 + 5, 4], [1e7 + 6, 5]])
    for chunks in [None, [slice(0, 3), slice(3, 6)]]:
        case(X, [0, 0, 0, 1, 1, 1], X[[1, 4]], dtype='float32',
             weights=[0.3, 1.7, 0.8, 1.1, 2.3, 0.2] if weighted else None,
             chunks=chunks, smoothing=0.2)
# Singular, absent-class, tiny-weight, explicit-prior and changing-epsilon paths.
for dtype in ['float64', 'float32']:
    for weights in [None, [1, 0, 0, 1], [1e-12, 1e-12, 1, 1]]:
        for smoothing in [0, 1e-9, 0.2]:
            case([[0, 0], [0, 0], [1, 3], [2, 7]], [9, 9, 2, 2], [[0, 0], [1, 4]],
                 dtype=dtype, weights=weights, smoothing=smoothing, classes=[9, -1, 2],
                 chunks=[slice(0, 2), slice(2, 4)])
    case([[0], [1], [4], [7]], [0, 0, 1, 1], [[2], [5]], priors=[0.2, 0.8], dtype=dtype)
    case([[0], [1], [4], [7]], [0, 0, 0, 0], [[2], [5]], dtype=dtype)
    for scale in [1e-10, 1e10]:
        case(np.array([[0, 1], [1, 2], [4, 3], [7, 4]]) * scale,
             [0, 0, 1, 1], np.array([[2, 1], [5, 4]]) * scale, dtype=dtype)
# Randomized dimensions and partitions in BOTH arithmetic dtypes. Include
# singleton batches, delayed rare classes, tiny/large scales and weighted fits.
for dtype in ('float32', 'float64'):
    for trial in range(10):
        n, f = int(rng.integers(7, 24)), int(rng.choice([1, 2, 7, 17]))
        X = rng.normal(size=(n, f)) * [1e-10, 1., 1e10][trial % 3]
        y = np.zeros(n, dtype=int)
        y[-2:] = [1, 2]
        cuts = sorted(set([0, 1, n - 1, n] + rng.integers(1, n, 3).tolist()))
        case(X, y, X[:3], dtype=dtype,
             weights=10. ** rng.uniform(-4, 4, n) if trial % 2 else None,
             chunks=[slice(a, b) for a, b in zip(cuts, cuts[1:])])
    for delta in (0., 1e-4, -1e-4):
        case([[-1], [1], [-1], [1]], [0, 0, 1, 1], [[0], [1e-5]],
             dtype=dtype, weights=[1, 1, 1, 1 + delta],
             chunks=[slice(0, 2), slice(2, 4)])
text = '\n'.join(lines) + '\n'
starts = list(re.finditer(r'@no_inline\ndef case_\d+\(\) raises:', text))
assert len(starts) == len(names)
prefix = text[:starts[0].start()]
executed = []
for first in range(0, len(names), 16):
    last = min(first + 16, len(names))
    end = starts[last].start() if last < len(starts) else len(text)
    batch = prefix + text[starts[first].start():end] + '\ndef main() raises:\n'
    batch += ''.join(f'    {name}()\n' for name in names[first:last])
    source = OUT / f'cases_{first}_{last}.mojo'
    binary = OUT / f'cases_{first}_{last}'
    source.write_text(batch)
    subprocess.run(['pixi', 'run', 'mojo', 'build', '-O3', '--fp-mode=contract=off',
                    '-I', str(ROOT / 'builds/deps'), str(source), '-o', str(binary)], cwd=ROOT, check=True)
    subprocess.run([str(binary)], check=True)
    executed.extend(names[first:last])
assert executed == names
print(f'PASS: {len(executed)} Gaussian sklearn differential cases', flush=True)
