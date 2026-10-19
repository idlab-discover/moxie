"""Dense incremental OvR parity; setup is shared with ovr_parity.py."""
import json
import subprocess

import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier

from ovr_parity import ROOT, OUT, REVISION, driver, lit, matrix, parse


class OnlineClassifier(ClassifierMixin, BaseEstimator):
    """Python twin of the native signed-sum test double, not an ML estimator port."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        for name in ('weights_', 'samples_', 'positives_', 'updates_'):
            if hasattr(self, name):
                delattr(self, name)
        return self.partial_fit(X, y, np.array([0, 1]))

    def partial_fit(self, X, y, classes=None):
        np.testing.assert_array_equal(classes, [0, 1])
        if not hasattr(self, 'weights_'):
            self.weights_ = [0.0] * X.shape[1]
            self.samples_ = self.positives_ = self.updates_ = 0
        assert X.shape[1] == len(self.weights_)
        for i in range(len(y)):
            assert y[i] in (0, 1)
            self.positives_ += int(y[i])
            for j in range(X.shape[1]):
                self.weights_[j] += (2.0 * y[i] - 1.0) * X[i, j]
        self.samples_ += len(y)
        self.updates_ += 1
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        out = []
        for row in X:
            value = self.shift
            for x, weight in zip(row, self.weights_):
                value += x * weight
            out.append(value)
        return np.array(out)

    def predict_proba(self, X):
        v = self.decision_function(X)
        p = 0.5 + 0.5 * v / (1.0 + abs(v))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def vector(values, dtype):
    return f"vec[DType.{dtype}]([{', '.join(lit(v) for v in values)}])"


def rejection_cases():
    """Exercise fresh calls, declarations, widths and nonincremental fit state."""
    from pathlib import Path
    import inspect
    from sklearn.preprocessing import LabelBinarizer
    assert Path(inspect.getfile(LabelBinarizer)).read_bytes() == (ROOT.parent / 'scikit-learn/sklearn/preprocessing/_label.py').read_bytes()
    X = np.array([[1.0], [2.0]])
    Y = np.array([[0, 1, 0], [1, 0, 1]])
    cases = []
    for initial in (None, np.array([0, 1]), Y, Y * 0, np.array([[0, 1, 0], [1, 0, 0]])):
        for y, classes in ((Y, None), (Y, [0, 1, 2]), (Y, [2, 3, 4]),
                           (Y[:, :2], None), (np.column_stack([Y, Y[:, 0]]), None),
                           (Y, [0, 1, 3]), (Y, [])):
            model = OneVsRestClassifier(OnlineClassifier())
            if initial is not None:
                model.fit(X, initial)
            try:
                model.partial_fit(X, y, classes)
            except (ValueError, AttributeError):
                cases.append((initial, y, classes))
    def target(y):
        return vector(y, 'int64') if y.ndim == 1 else f'mat({matrix(y)}).astype[DType.int64]()'
    source = '\nfrom std.testing import assert_raises, assert_equal\n\ndef rejection_cases() raises:\n'
    for i, (initial, y, classes) in enumerate(cases):
        source += f'    var m{i} = OneVsRestClassifier(OnlineClassifier())\n'
        if initial is not None:
            source += f'    m{i}.fit(mat({matrix(X)}), {target(initial)})\n'
        arg = '' if classes is None else ', ' + vector(classes, 'int64')
        source += f'    with assert_raises():\n        m{i}.partial_fit(mat({matrix(X)}), {target(y)}{arg})\n'
    # zip drops a trailing constant on a two-label vector transition.
    initial = np.array([[0, 0], [1, 0]])
    model = OneVsRestClassifier(OnlineClassifier()).fit(X, initial)
    model.partial_fit(X, np.array([0, 1]))
    assert len(model.estimators_) == 1 and model.estimators_[0].updates_ == 2
    source += "    var trailing = OneVsRestClassifier(OnlineClassifier())\n"
    source += f"    trailing.fit(mat({matrix(X)}), {target(initial)})\n"
    source += f"    trailing.partial_fit(mat({matrix(X)}), {vector([0, 1], 'int64')})\n"
    source += "    assert_equal(len(trailing.estimators_), 1)\n"
    source += "    assert_equal(trailing.estimators_[0].estimator().updates, 2)\n"
    return source, len(cases)


def main():
    # Importing ovr_parity verifies the local revision and installed source bytes.
    source = driver([]).split('def main() raises:')[0]
    source += '\nfrom ovr_incremental_fixtures import OnlineClassifier\n'
    expected, names = [], []
    functions = []
    rng = np.random.default_rng(20261004)
    for k in (1, 2, 3, 5):
        for dtype in ('int64', 'float32', 'float64'):
            for jobs in (1, 2, -1):
                for start_fit in (False, True, "multilabel"):
                    if start_fit and k == 1:
                        continue  # ordinary fit creates a nonincremental constant
                    pool = np.array([-7, 0, 3, 9, 12][:k], dtype=dtype)
                    multilabel = start_fit == "multilabel"
                    if multilabel:
                        pool = np.arange(k, dtype=dtype)
                    model = OneVsRestClassifier(OnlineClassifier(shift=0.25), n_jobs=jobs)
                    idx = len(functions)
                    fn = f'case_{idx}'
                    functions.append(fn)
                    lines = [f'\n@no_inline\ndef {fn}() raises:',
                             f'    var m = OneVsRestClassifier[OnlineClassifier, label=DType.{dtype}](OnlineClassifier(shift=0.25), n_jobs={jobs})']
                    Xt = rng.integers(-3, 4, size=(7, 2)).astype(float) / 4
                    lines.append(f'    var Xt = mat({matrix(Xt)})')
                    for batch in range(5 if multilabel else 3):
                        y = (pool.copy() if start_fit and batch == 0 else
                             np.repeat(pool[-1], 3) if batch == 0 else
                             rng.choice(pool, size=5))
                        if multilabel and batch != 3:
                            y = (np.array([[j % 2 for j in range(k)], [(j + 1) % 2 for j in range(k)]], dtype=dtype)
                                 if batch in (0, 4) else np.full((3, k), batch - 1, dtype=dtype))
                        X = rng.integers(-5, 6, size=(len(y), 2)).astype(float) / 4
                        lines += [f'    var X{batch} = mat({matrix(X)})',
                                  f'    var y{batch} = ' + (f'mat({matrix(y)}).astype[DType.{dtype}]()' if y.ndim == 2 else vector(y, dtype))]
                        # Include column-vector targets; classes stay rank one.
                        target = f'y{batch}'
                        if batch == 1 and not multilabel:
                            target += f'.reshape(Shape({len(y)}, 1))'
                            y = y.reshape(-1, 1)
                        if start_fit and batch == 0:
                            model.fit(X, y)
                            lines.append(f'    m.fit(X{batch}, {target})')
                        else:
                            classes = np.concatenate([pool[::-1], pool[:1]]) if batch != 1 else None
                            model.partial_fit(X, y, classes=classes)
                            arg = '' if classes is None else ', ' + vector(classes, dtype)
                            lines.append(f'    m.partial_fit(X{batch}, {target}{arg})')
                        out = dict(classes=model.classes_,
                                   decision=model.decision_function(Xt), proba=model.predict_proba(Xt),
                                   n_estimators=len(model.estimators_), multilabel=int(model.multilabel_),
                                   updates=[e.updates_ for e in model.estimators_],
                                   samples=[e.samples_ for e in model.estimators_],
                                   positives=[e.positives_ for e in model.estimators_])
                        try:
                            out['predict'] = model.predict(Xt)
                        except ValueError:
                            assert multilabel and k == 2 and batch >= 3
                        expected.append(out)
                        names.append(f'k{k}-{dtype}-jobs{jobs}-fit{start_fit}-batch{batch}')
                        lines += ["    print('{')", "    emit('classes', m.classes_)",
                                  "    emit('decision', m.decision_function(Xt))",
                                  "    emit('proba', m.predict_proba(Xt))",
                                  "    print('\"n_estimators\":', len(m.estimators_), ',')",
                                  "    print('\"multilabel\":', Int(m.multilabel_()), ',')"]
                        if 'predict' in out:
                            lines.append("    emit('predict', m.predict(Xt))")
                        else:
                            lines += ["    with assert_raises():", "        _ = m.predict(Xt)"]
                        for attr in ('updates', 'samples', 'positives'):
                            lines += [f'    var {attr}{batch} = nm.zeros[DType.int64](Shape(len(m.estimators_)))',
                                      '    for j in range(len(m.estimators_)):',
                                      f'        {attr}{batch}[Item(j)] = Int64(m.estimators_[j].estimator().{attr})',
                                      f"    emit('{attr}', {attr}{batch})"]
                        lines.append("    print('\"end\": 0}')")
                    source += '\n'.join(lines) + '\n'
    rejection_source, rejections = rejection_cases()
    source += rejection_source
    functions.append('rejection_cases')
    source += '\ndef main() raises:\n' + ''.join(f'    {f}()\n' for f in functions)
    path = OUT / 'ovr_partial_fit_driver.mojo'
    path.write_text(source)
    binary = OUT / 'ovr_partial_fit_driver'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT / 'builds/deps',
                    '-I', ROOT / 'tests/multiclass', path, '-o', binary], check=True)
    actual = parse(subprocess.check_output([binary], text=True))
    assert len(actual) == len(expected)
    failures = []
    for name, want, got in zip(names, expected, actual):
        for key, value in want.items():
            a, b = np.asarray(value), np.asarray(got[key])
            # Quarter-grid fixture arithmetic is exact; only probability division
            # and row normalization require rounding tolerance.
            ok = a.shape == b.shape and (np.allclose(a, b, rtol=1e-14, atol=1e-15)
                                         if key == 'proba' else np.array_equal(a, b))
            if not ok:
                failures.append((name, key, a.tolist(), b.tolist()))
    (OUT / 'ovr-partial-fit-parity-report.json').write_text(json.dumps(
        dict(revision=REVISION, checkpoints=len(expected), rejections=rejections, failures=failures), indent=2))
    assert not failures, failures
    print(f'PASS: {len(expected)} dense OvR incremental checkpoints and {rejections} rejections match pinned sklearn')


if __name__ == '__main__':
    with parallel_backend('threading'):
        main()
