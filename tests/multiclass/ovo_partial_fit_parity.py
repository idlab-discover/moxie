"""Dense incremental OvO against the pinned sklearn; shared online test doubles.

Run with builds/reference-env/bin/python under pixi, after pixi run build.
Quarter-grid arithmetic makes fixture updates exact; confidence normalization
allows 1e-14 relative / 1e-15 absolute for floating division rounding.
"""
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
from joblib import parallel_backend
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils import multiclass as labels_module
from sklearn.utils import validation

from ovr_parity import ROOT, OUT, REVISION, driver, matrix, parse
from ovr_partial_fit_parity import OnlineClassifier, vector

# The imported shared driver verifies multiclass.py and HEAD; also pin helpers.
for module in (labels_module, validation):
    relative = 'sklearn/utils/' + Path(inspect.getfile(module)).name
    pinned = subprocess.check_output(
        ['git', '-C', ROOT.parent / 'scikit-learn', 'show', f'{REVISION}:{relative}'])
    assert Path(inspect.getfile(module)).read_bytes() == pinned


def rejection_cases():
    X = np.array([[1.0], [2.0]])
    y = np.array([2.0, 7.0])
    cases = [
        (X, y, None), (X, y, []), (X, [2, 2], [2]),
        (X, y, [2, 3]), (X, y, [2.5, 7]),
        (X, y, [2, np.nan]), (X, y, [2, np.inf]),
        (X, [2, 9], [2, 7]), (X, [2.5, 7], [2, 7]),
        (X, [np.nan, 7], [2, 7]), (X, [np.inf, 7], [2, 7]),
        (X, [2], [2, 7]), (X, [], [2, 7]),
        (X.ravel(), y, [2, 7]), (X.reshape(1, 2, 1), y, [2, 7]),
        (np.empty((0, 1)), [], [2, 7]), (np.empty((2, 0)), y, [2, 7]),
        (X, [[2, 7], [7, 2]], [2, 7]),
    ]
    source = '\ndef rejection_cases() raises:\n'
    count = 0
    def array(a):
        a = np.asarray(a, dtype=float)
        dims = ', '.join(map(str, a.shape))
        return vector(a.ravel(), 'float64').replace('inf', 'inf[DType.float64]()') + f'.reshape(Shape({dims}))'
    for initial in ('fresh', 'fit', 'partial'):
        for bad_X, bad_y, classes in cases:
            m = OneVsOneClassifier(OnlineClassifier())
            if initial == 'fit':
                m.fit(X, y)
            elif initial == 'partial':
                m.partial_fit(X, y, [2, 7])
            try:
                m.partial_fit(bad_X, bad_y, classes)
            except (ValueError, IndexError):
                pass
            else:
                continue
            name = f'm{count}'
            source += f'    var {name} = OneVsOneClassifier[OnlineClassifier, label=DType.float64](OnlineClassifier())\n'
            if initial != 'fresh':
                method = 'fit' if initial == 'fit' else 'partial_fit'
                arg = '' if initial == 'fit' else ', ' + array(y)
                source += f'    {name}.{method}({array(X)}, {array(y)}{arg})\n'
            arg = '' if classes is None else ', ' + array(classes)
            source += f'    with assert_raises():\n        {name}.partial_fit({array(bad_X)}, {array(bad_y)}{arg})\n'
            count += 1
        for bad_X, classes, jobs in ((np.ones((2, 2)), None, 1), (X, [2, 7, 9], 1), (X, None, 0)):
            m = OneVsOneClassifier(OnlineClassifier(), n_jobs=jobs)
            m.partial_fit(X, y, [2, 7]) if jobs else None
            if jobs == 0:
                classes = [2, 7]
            try:
                m.partial_fit(bad_X, y, classes)
            except ValueError:
                pass
            else:
                raise AssertionError('expected rejection')
            name = f'm{count}'
            source += f'    var {name} = OneVsOneClassifier[OnlineClassifier, label=DType.float64](OnlineClassifier(), n_jobs={jobs})\n'
            if jobs:
                source += f'    {name}.partial_fit({array(X)}, {array(y)}, {array(y)})\n'
            arg = '' if classes is None else ', ' + array(classes)
            source += f'    with assert_raises():\n        {name}.partial_fit({array(bad_X)}, {array(y)}{arg})\n'
            count += 1
    return source, count


def main():
    source = driver([]).split('def main() raises:')[0]
    source += '\nfrom moxie.multiclass import OneVsOneClassifier\nfrom ovr_incremental_fixtures import OnlineClassifier\nfrom std.testing import assert_raises\nfrom std.math import inf\n'
    helpers = """
from moxie.base import IncrementalClassifier, SupervisedEstimator

@no_inline
def update[E: IncrementalClassifier](mut m: E,
    X: E.Input, y: E.Target, classes: Optional[E.Classes] = None) raises:
    _ = m.partial_fit(X, y, classes)

@no_inline
def fit_model[E: SupervisedEstimator](mut m: E,
    X: E.Input, y: E.Target) raises:
    _ = m.fit(X, y)

"""
    source += helpers
    expected, names, functions = [], [], []
    rng = np.random.default_rng(20261004)
    for k in (2, 3, 5):
        for dtype in ('int64', 'float32', 'float64'):
            for jobs in (1, 2, -1):
                for start_fit in (False, True):
                    pool = np.array([-7, 0, 3, 9, 12][:k], dtype=dtype)
                    model = OneVsOneClassifier(OnlineClassifier(shift=0.25), n_jobs=jobs)
                    fn = f'case_{len(functions)}'
                    functions.append(fn)
                    lines = [f'\n@no_inline\ndef {fn}() raises:',
                             f'    var m = OneVsOneClassifier[OnlineClassifier, label=DType.{dtype}](OnlineClassifier(shift=0.25), n_jobs={jobs})']
                    Xt = rng.integers(-3, 4, size=(7, 2)).astype(float) / 4
                    lines.append(f'    var Xt = mat({matrix(Xt)})')
                    for batch in range(6):
                        y = (pool.copy() if (start_fit and batch == 0) or batch in (2, 4)
                             else np.repeat(pool[-1], 3) if batch in (0, 3)
                             else np.repeat(pool[0], 2) if batch == 1
                             else rng.choice(pool, size=7))
                        X = rng.integers(-5, 6, size=(len(y), 2)).astype(float) / 4
                        lines += [f'    var X{batch} = mat({matrix(X)})',
                                  f'    var y{batch} = {vector(y, dtype)}']
                        target = f'y{batch}'
                        if batch == 1:
                            target += f'.reshape(Shape({len(y)}, 1))'
                            y = y.reshape(-1, 1)
                        if (start_fit and batch == 0) or batch == 4:
                            model.fit(X, y)
                            lines.append(f'    fit_model(m, X{batch}, {target})')
                        else:
                            classes = np.concatenate([pool[::-1], pool[:1]]) if batch in (0, 2, 5) else None
                            model.partial_fit(X, y, classes)
                            arg = '' if classes is None else ', ' + vector(classes, dtype)
                            lines.append(f'    update(m, X{batch}, {target}{arg})')
                        out = dict(classes=model.classes_, n_estimators=len(model.estimators_),
                                   n_features=model.n_features_in_, pairwise=int(model.pairwise_indices_ is None),
                                   updates=[getattr(e, 'updates_', 0) for e in model.estimators_],
                                   samples=[getattr(e, 'samples_', 0) for e in model.estimators_],
                                   positives=[getattr(e, 'positives_', 0) for e in model.estimators_])
                        lines += ["    print('{')", "    emit('classes', m.classes_)",
                                  "    print('\"n_estimators\":', len(m.estimators_), ',')",
                                  "    print('\"n_features\":', m.n_features_in_.value(), ',')",
                                  "    print('\"pairwise\":', Int(not m.pairwise_indices_), ',')"]
                        for method in ('predict', 'decision_function'):
                            try:
                                out[method] = getattr(model, method)(Xt)
                            except AttributeError:
                                assert 0 in out['updates']
                                lines += ['    with assert_raises():', f'        _ = m.{method}(Xt)']
                            else:
                                lines.append(f"    emit('{method}', m.{method}(Xt))")
                        for attr in ('updates', 'samples', 'positives'):
                            lines += [f'    var {attr}{batch} = nm.zeros[DType.int64](Shape(len(m.estimators_)))',
                                      '    for j in range(len(m.estimators_)):',
                                      f'        {attr}{batch}[Item(j)] = Int64(m.estimators_[j].{attr})',
                                      f"    emit('{attr}', {attr}{batch})"]
                        lines.append("    print('\"end\": 0}')")
                        expected.append(out)
                        names.append(f'k{k}-{dtype}-jobs{jobs}-fit{start_fit}-batch{batch}')
                    source += '\n'.join(lines) + '\n'
    rejections, count = rejection_cases()
    source += rejections
    functions.append('rejection_cases')
    source += '\ndef main() raises:\n' + ''.join(f'    {fn}()\n' for fn in functions)
    path = OUT / 'ovo_incremental_parity.mojo'
    path.write_text(source)
    binary = OUT / 'ovo_incremental_parity'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT / 'builds/deps',
                    '-I', ROOT / 'tests/multiclass', path, '-o', binary], check=True)
    actual = parse(subprocess.check_output([binary], text=True))
    assert len(actual) == len(expected)
    failures = []
    for name, want, got in zip(names, expected, actual):
        for key, value in want.items():
            a, b = np.asarray(value), np.asarray(got[key])
            ok = a.shape == b.shape and (np.allclose(a, b, rtol=1e-14, atol=1e-15)
                                         if key == 'decision_function' else np.array_equal(a, b))
            if not ok:
                failures.append((name, key, a.tolist(), b.tolist()))
    (OUT / 'ovo-partial-fit-parity-report.json').write_text(json.dumps(
        dict(revision=REVISION, checkpoints=len(expected), rejections=count, failures=failures), indent=2))
    assert not failures, failures
    print(f'PASS: {len(expected)} dense OvO incremental checkpoints and {count} rejections match pinned sklearn')


if __name__ == '__main__':
    with parallel_backend('threading'):
        main()
