"""Pinned sklearn parity for string domains and all three wrappers.

Labels, classes, counts and predictions compare exactly; decisions/probabilities
use 5e-11 relative, 1e-13 absolute tolerance for fixture arithmetic. Weighted
scores allow 1e-15. Includes ordinary and incremental fitting.
"""
import json
import subprocess

import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from joblib import parallel_backend

from ovr_parity import OVR_COMPONENTS, OUT, ROOT, REVISION, driver, matrix, parse
from ovr_partial_fit_parity import OnlineClassifier

class StringOnlineClassifier(OnlineClassifier):
    """Normalize sklearn OvO's string-coded binary targets to Moxie's int64 ABI."""
    def partial_fit(self, X, y, classes=None):
        if np.asarray(y).dtype.kind in 'US':
            assert set(y) <= {'', '1'}
            y = (np.asarray(y) == '1').astype(np.int64)
        return super().partial_fit(X, y, classes)


WRAPPERS = {c.__name__: c for c in (OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier)}
DOMAIN = np.array(['猫', '', 'é', 'alpha', '😀'])


def cases():
    rng = np.random.default_rng(20261004)
    result = []
    for wrapper in WRAPPERS:
        for component in ('MarginClassifier', 'ProbaClassifier', 'MarginProbaClassifier', 'DecisionRegressor'):
            for count in (1, 2, 3, 5):
                if count == 1 and wrapper == 'OneVsOneClassifier':
                    continue
                for jobs in (1, 2):
                    y = np.tile(DOMAIN[:count], 3)
                    rng.shuffle(y)
                    result.append(dict(wrapper=wrapper, component=component, jobs=jobs,
                                       X=rng.normal(size=(len(y), 2)), y=y,
                                       Xt=rng.normal(size=(4, 2)), column=jobs == 2, incremental=False))
        for jobs in (1, 2):
            result.append(dict(wrapper=wrapper, component='MarginProbaClassifier', jobs=jobs,
                               X=np.zeros((3, 2)), y=DOMAIN[:3], Xt=np.zeros((2, 2)),
                               column=False, incremental=False))
        if wrapper != 'OutputCodeClassifier':
            for jobs in (1, 2):
                result.append(dict(wrapper=wrapper, component='OnlineClassifier', jobs=jobs,
                                   X=rng.normal(size=(6, 2)), y=np.tile(DOMAIN[:3], 2),
                                   Xt=rng.normal(size=(4, 2)), column=False, incremental=True))
    return result


def reference(c):
    component = StringOnlineClassifier if c['incremental'] else OVR_COMPONENTS[c['component']]
    kwargs = dict(n_jobs=c['jobs'])
    if c['wrapper'] == 'OutputCodeClassifier':
        kwargs.update(random_state=17, code_size=2)
    m = WRAPPERS[c['wrapper']](component(), **kwargs)
    y = c['y'].reshape(-1, 1) if c['column'] else c['y']
    with parallel_backend('threading'):
        if c['incremental']:
            m.partial_fit(c['X'], y, classes=DOMAIN[:3])
            m.partial_fit(c['X'], y)
        else:
            m.fit(c['X'], y)
        result = dict(classes=m.classes_.tolist(), predict=m.predict(c['Xt']).tolist(),
                      n_estimators=len(m.estimators_), score=m.score(c['X'], y),
                      wscore=m.score(c['X'], y, sample_weight=np.arange(1, len(y)+1)))
        if hasattr(m, 'decision_function') and (c['wrapper'] != 'OneVsRestClassifier' or hasattr(component, 'decision_function')):
            result['decision'] = m.decision_function(c['Xt']).tolist()
        if hasattr(m, 'predict_proba') and hasattr(component, 'predict_proba'):
            result['proba'] = m.predict_proba(c['Xt']).tolist()
        if hasattr(m, 'code_book_'):
            result['book'] = m.code_book_.tolist()
    return result


def string_literal(values):
    return '[' + ', '.join(json.dumps(str(v), ensure_ascii=False) for v in values) + ']'


def native_source(all_cases, expected):
    header = driver([]).split('def main() raises:')[0]
    header += '''
from moxie.multiclass import OneVsOneClassifier, OutputCodeClassifier, StringLabels, StringTarget
from ovr_incremental_fixtures import OnlineClassifier

# Corpus strings contain no JSON control characters or quotes.
def emit(name: String, values: List[String]):
    var line = String('"') + name + '": ['
    for i in range(len(values)):
        line += (", " if i else "") + String('"') + values[i] + '"'
    print(line + "],")

def emit(name: String, y: StringTarget):
    emit(name, y.values())

'''
    body = []
    for i, (c, ref) in enumerate(zip(all_cases, expected)):
        target = string_literal(c['y'])
        if c['column']:
            target += f", {len(c['y'])}, 1"
        params = f"n_jobs={c['jobs']}"
        if c['wrapper'] == 'OutputCodeClassifier':
            params += ', random_state=17, code_size=2'
        body += [f'def case_{i}() raises:', f'    var X = mat({matrix(c["X"])})',
                 f'    var Xt = mat({matrix(c["Xt"])})', f'    var y = StringTarget({target})',
                 f'    var m = {c["wrapper"]}[{c["component"]}, kind=StringLabels]({c["component"]}(), {params})']
        if c['incremental']:
            body += [f'    m.partial_fit(X, y, StringTarget({string_literal(DOMAIN[:3])}))',
                     '    m.partial_fit(X, y)']
        else:
            body += ['    m.fit(X, y)']
        body += [f'    var w = vec[DType.float64]([{", ".join(str(j) for j in range(1, len(c["y"])+1))}])',
                 f'    print(\'{{"case": {i},\')', '    emit("classes", m.classes_)',
                 '    emit("predict", m.predict(Xt))',
                 '    print(\'"n_estimators":\', len(m.estimators_), ",")',
                 '    print(\'"score":\', m.score(X, y), ",")',
                 '    print(\'"wscore":\', m.score(X, y, w.copy()), ",")']
        for key in ('decision', 'proba', 'book'):
            if key in ref:
                expr = dict(decision='m.decision_function(Xt)', proba='m.predict_proba(Xt)', book='m.code_book_')[key]
                body += [f'    emit("{key}", {expr})']
        body += ['    print(\'"end": 0}\')', '']
    return header + '\n'.join(body + ['def main() raises:'] + [f'    case_{i}()' for i in range(len(all_cases))]) + '\n'


def main():
    all_cases = cases()
    expected = [reference(c) for c in all_cases]
    source, binary = OUT / 'string_labels_driver.mojo', OUT / 'string_labels_driver'
    source.write_text(native_source(all_cases, expected))
    command = ['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT/'builds/deps', '-I', ROOT/'tests/multiclass', source, '-o', binary]
    subprocess.run(command, check=True)
    actual = parse(subprocess.run([binary], capture_output=True, check=True, text=True).stdout)
    assert len(actual) == len(expected)
    failures = []
    for i, (ref, got) in enumerate(zip(expected, actual)):
        for key in ref:
            if key in ('classes', 'predict', 'n_estimators', 'book', 'score'):
                ok = np.array_equal(ref[key], got[key])
            else:
                tolerance = 1e-15 if key == 'wscore' else 5e-11
                ok = np.shape(ref[key]) == np.shape(got[key]) and np.allclose(ref[key], got[key], rtol=tolerance, atol=1e-15 if key == 'wscore' else 1e-13)
            if not ok:
                failures.append([i, key])
    report = dict(revision=REVISION, cases=len(all_cases), failures=failures, command=list(map(str, command)))
    (OUT/'string-labels-report.json').write_text(json.dumps(report, indent=2)+'\n')
    (OUT/'string-labels-expected.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2)+'\n')
    (OUT/'string-labels-actual.json').write_text(json.dumps(actual, ensure_ascii=False, indent=2)+'\n')
    assert not failures, failures
    print(f'PASS: {len(all_cases)} string-label cases match pinned sklearn')


if __name__ == '__main__':
    main()
