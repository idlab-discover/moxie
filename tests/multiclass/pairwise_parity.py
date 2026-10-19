"""Pinned sklearn fixtures for dense precomputed multiclass input routing.

Python and Mojo wrap matching recording test doubles, not ported kernel learners.
Seeded dyadic inputs make gathers and signed sums exact in both feature dtypes.
Decision/probability divisions allow 1e-14 relative / 1e-15 absolute rounding;
indices, targets, gathered matrices, labels, and predictions must match exactly.
"""
from pathlib import Path
import inspect
import json
import subprocess

import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.utils import get_tags
from sklearn.utils import metaestimators

from ovr_parity import OUT, ROOT, REFERENCE, REVISION, matrix, lit, parse
from ovo_parity import driver
from string_labels_parity import string_literal

# Source-byte pin extends the common multiclass pin.
assert Path(inspect.getfile(metaestimators)).read_bytes() == (REFERENCE / 'sklearn/utils/metaestimators.py').read_bytes()


class Recorder(ClassifierMixin, BaseEstimator):
    def __init__(self, precomputed=True, fail_rows=-1):
        self.precomputed = precomputed
        self.fail_rows = fail_rows

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.precomputed
        return tags

    def fit(self, X, y):
        if self.precomputed and X.shape[0] != X.shape[1]:
            raise ValueError('fixture requires a square kernel')
        self.weights = []
        self.updates = 0
        return self._update(X, y)

    def _update(self, X, y):
        if X.shape[0] == self.fail_rows:
            raise ValueError('intentional pair failure')
        if getattr(self, 'updates', 0) and X.shape[1] != len(self.weights):
            raise ValueError('changed feature count')
        if not getattr(self, 'updates', 0):
            self.weights = [0.0] * X.shape[1]
            self.updates = 0
        self.seen = X.copy()
        self.targets = np.asarray(y).copy()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        for i in range(len(y)):
            sign = 1.0 if y[i] == 1 else -1.0
            for j in range(X.shape[1]):
                self.weights[j] += sign * float(X[i, j])
        self.updates += 1
        return self

    def partial_fit(self, X, y, classes=None):
        assert np.array_equal(classes, [0, 1])
        return self._update(X, y)

    def margin(self, X):
        if X.shape[1] != len(self.weights):
            raise ValueError('changed feature count')
        result = []
        for row in X:
            value = 0.0
            for j, x in enumerate(row):
                value += float(x) * self.weights[j]
            result.append(value)
        return np.array(result)

    def predict(self, X):
        return (self.margin(X) > 0).astype(np.int64)


class MarginRecorder(Recorder):
    def decision_function(self, X):
        return self.margin(X)


class ProbaRecorder(Recorder):
    def predict_proba(self, X):
        values = self.margin(X)
        p = 0.5 + 0.5 * values / (1 + abs(values))
        return np.column_stack([1-p, p])


WRAPPERS = {c.__name__: c for c in (OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier)}


def cases():
    rng = np.random.default_rng(20261005)
    result = []
    for count in (2, 3, 5):
        # Unbalanced and interleaved classes, including singleton class 4.
        codes = np.arange(11) % count
        if count == 5:
            codes[-1] = 1
            codes[9] = 0
        rng.shuffle(codes)
        features = rng.integers(-8, 9, (11, 4)).astype(float) / 8
        test = rng.integers(-8, 9, (4, 4)).astype(float) / 8
        for dtype in ('float32', 'float64'):
            X = (features @ features.T).astype(dtype)
            Xt = (test @ features.T).astype(dtype)
            for label in ('int64', 'float64', 'string'):
                pool = np.array(['猫', '', 'é', 'alpha', '😀']) if label == 'string' else np.array([7, -3, 11, 2, 0], dtype=label)
                for probability in (False, True):
                    for jobs in (1, 2):
                        result.append(dict(wrapper='OneVsOneClassifier', X=X, Xt=Xt, y=pool[codes],
                                           label=label, dtype=dtype, probability=probability, jobs=jobs,
                                           incremental=False))
    # Full-input forwarding for OvR and OutputCode, and row-only incremental OvO.
    for wrapper in WRAPPERS:
        for probability in (False, True):
            for jobs in (1, 2):
                c = result[0].copy()
                c.update(wrapper=wrapper, probability=probability, jobs=jobs,
                         incremental=wrapper == 'OneVsOneClassifier')
                c['y'] = np.array([7, -3, 2, 7, -3, 2, 7, -3, 2, 7, 7])
                result.append(c)
    return result


def reference(c):
    options = dict(n_jobs=c['jobs'])
    if c['wrapper'] == 'OutputCodeClassifier':
        options.update(random_state=17, code_size=2)
    component = ProbaRecorder if c['probability'] else MarginRecorder
    m = WRAPPERS[c['wrapper']](component(), **options)
    with parallel_backend('threading'):
        if c['incremental']:
            m.partial_fit(c['X'], c['y'], classes=np.unique(c['y']))
            m.partial_fit(c['X'], c['y'])
        else:
            m.fit(c['X'], c['y'])
    out = dict(classes=m.classes_.tolist(), predict=m.predict(c['Xt']).tolist(),
               pairwise=get_tags(m).input_tags.pairwise, n_estimators=len(m.estimators_),
               score=m.score(c['X'], c['y']),
               wscore=m.score(c['X'], c['y'], sample_weight=np.arange(1, len(c['y'])+1)))
    if c['wrapper'] == 'OneVsOneClassifier':
        out['width'] = m.n_features_in_
        out['indices_none'] = m.pairwise_indices_ is None
        if m.pairwise_indices_ is not None:
            for k, ix in enumerate(m.pairwise_indices_):
                out[f'indices{k}'] = ix.tolist()
    for k, e in enumerate(m.estimators_):
        if hasattr(e, 'seen'):
            out[f'seen{k}'] = e.seen.tolist()
            out[f'targets{k}'] = e.targets.tolist()
            out[f'updates{k}'] = e.updates
    if c['wrapper'] == 'OneVsOneClassifier' or (c['wrapper'] == 'OneVsRestClassifier' and not c['probability']):
        out['decision'] = m.decision_function(c['Xt']).tolist()
    if c['wrapper'] == 'OneVsRestClassifier' and c['probability']:
        out['proba'] = m.predict_proba(c['Xt']).tolist()
    return out


def native_source(all_cases, expected):
    header = driver([]).split('def main() raises:')[0]
    header += '''
from moxie.multiclass import OneVsRestClassifier, OutputCodeClassifier, StringLabels, StringTarget, pairwise_input
from pairwise_fixtures import PairwiseRecorder

def emit(name: String, values: List[String]):
    var line = String('"') + name + '": ['
    for i in range(len(values)):
        line += (", " if i else "") + String('"') + values[i] + '"'
    print(line + "],")

def emit(name: String, y: StringTarget):
    emit(name, y.values())
'''
    body = []
    for idx, (c, expected_case) in enumerate(zip(all_cases, expected)):
        dtype = f"DType.{c['dtype']}"
        E = f"PairwiseRecorder[{dtype}, probability={c['probability']}]"
        label_arg = ', kind=StringLabels' if c['label'] == 'string' else f", label=DType.{c['label']}"
        y = f"StringTarget({string_literal(c['y'])})" if c['label'] == 'string' else f"vec[DType.{c['label']}]([{', '.join(lit(v) for v in c['y'])}])"
        options = f"n_jobs={c['jobs']}" + (', random_state=17, code_size=2' if c['wrapper'] == 'OutputCodeClassifier' else '')
        lines = [f"var X = mat({matrix(c['X'])}).astype[{dtype}]()",
                 f"var Xt = mat({matrix(c['Xt'])}).astype[{dtype}]()",
                 f"var y = {y}", f"var m = {c['wrapper']}[{E}, dtype={dtype}{label_arg}]({E}(), {options})"]
        if c['incremental']:
            lines += ['m.partial_fit(X, y, y.copy())', 'm.partial_fit(X, y)']
        else:
            lines += ['m.fit(X, y)']
        lines += [f"print('{{\"case\": {idx},')", "emit('classes', m.classes_)", "emit('predict', m.predict(Xt))",
                  '''print('"pairwise":', 'true' if pairwise_input(m) else 'false', ',')''',
                  '''print('"n_estimators":', len(m.estimators_), ',')''',
                  '''print('"score":', m.score(X, y), ',')''',
                  f"var w = vec[DType.float64]([{', '.join(lit(v) for v in range(1, len(c['y'])+1))}])",
                  '''print('"wscore":', m.score(X, y, w.copy()), ',')''']
        if c['wrapper'] == 'OneVsOneClassifier':
            lines += ['''print('"width":', m.n_features_in_.value(), ',')''',
                      '''print('"indices_none":', 'false' if m.pairwise_indices_ else 'true', ',')''']
            for k in range(expected_case['n_estimators']):
                if f'indices{k}' in expected_case:
                    lines += [f"emit('indices{k}', m.pairwise_indices_.value()[{k}])"]
        for k in range(expected_case['n_estimators']):
            if f'seen{k}' not in expected_case:
                continue
            e = f'm.estimators_[{k}]' + ('' if c['wrapper'] == 'OneVsOneClassifier' else '.estimator()')
            lines += [f"emit('seen{k}', {e}.seen)", f"emit('targets{k}', {e}.targets)",
                      f'''print('"updates{k}":', {e}.updates, ',')''']
        if 'decision' in expected_case:
            lines += ["emit('decision', m.decision_function(Xt))"]
        if 'proba' in expected_case:
            lines += ["emit('proba', m.predict_proba(Xt))"]
        lines += ['''print('"end": 0}')''']
        body += [f'@no_inline\ndef case_{idx}() raises:\n' + '\n'.join('    ' + s for s in lines)]
    body += ['def main() raises:\n' + '\n'.join(f'    case_{i}()' for i in range(len(all_cases)))]
    return header + '\n\n'.join(body) + '\n'


def audit():
    X = np.arange(36).reshape(6, 6).astype(float)
    y = np.array([0, 1, 2, 0, 1, 2])
    m = OneVsOneClassifier(MarginRecorder())
    try:
        m.fit(X[:, :5], y)
    except ValueError as e:
        assert str(e) == 'X should be a square kernel matrix'
    else:
        raise AssertionError('rectangular batch kernel accepted')
    m.fit(X, y)
    try:
        m.predict(X[:, :5])
    except ValueError as e:
        assert str(e) == 'X has 5 features, but OneVsOneClassifier is expecting 6 features as input.'
    else:
        raise AssertionError('wrong prediction width accepted')
    # Incremental input is rectangular and selects only rows, even with pairwise=True.
    m = OneVsOneClassifier(MarginRecorder()).partial_fit(X[:, :5], y, classes=[0, 1, 2])
    assert m.pairwise_indices_ is None
    assert all(e.seen.shape == (4, 5) for e in m.estimators_)
    m.fit(X, y)
    try:
        m.partial_fit(X, y)
    except ValueError as e:
        assert str(e) == 'changed feature count'
    else:
        raise AssertionError('batch component unexpectedly accepted full-width incremental input')
    m.fit(X, y % 2).partial_fit(X[:1], y[:1] % 2)
    assert m.pairwise_indices_ is None


def main():
    audit()
    all_cases = cases()
    expected = [reference(c) for c in all_cases]
    source = OUT / 'pairwise_driver.mojo'
    source.write_text(native_source(all_cases, expected))
    binary = OUT / 'pairwise_driver'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT/'builds/deps',
                    '-I', ROOT/'tests/multiclass', source, '-o', binary], check=True)
    actual = parse(subprocess.check_output([binary], text=True))
    assert len(actual) == len(expected)
    worst = 0.0
    for idx, (want, got) in enumerate(zip(expected, actual)):
        for key, value in want.items():
            if key in ('decision', 'proba', 'score', 'wscore'):
                a, b = np.asarray(value), np.asarray(got[key])
                assert a.shape == b.shape, (idx, key)
                np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-15, err_msg=f'{idx}: {key}')
                worst = max(worst, float(np.max(abs(a-b))))
            else:
                assert got[key] == value, (idx, key, value, got[key])
    report = dict(revision=REVISION, cases=len(all_cases), max_absolute_error=worst,
                  rtol=1e-14, atol=1e-15, audit='batch/partial_fit/OvR/OutputCode passed')
    (OUT/'pairwise-fixtures.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2))
    (OUT/'pairwise-parity-report.json').write_text(json.dumps(report, indent=2))
    print(f'PASS: {len(all_cases)} dense pairwise cases; max absolute error {worst}; incremental/tag audit passed')


if __name__ == '__main__':
    main()
