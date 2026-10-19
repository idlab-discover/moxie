"""Compare OutputCodeClassifier to the pinned local sklearn, using twin components.

Setup is shared with ovr_parity.py. Run `pixi run parity`, or after building:
    pixi run builds/reference-env/bin/python tests/multiclass/output_code_parity.py

Codebooks, labels, constants, counts, predictions and unweighted scores must
match exactly. Component confidence tolerances cover only fixture arithmetic:
1e-12 relative/1e-14 absolute for margins, 5e-11 relative for std.math.exp.
The oracle is sklearn's original estimator, including its distance decoder.
"""
import json
import math
import subprocess

import numpy as np
import sklearn
from sklearn.multiclass import OutputCodeClassifier, _predict_binary

from ovr_parity import COMPONENTS as CLASSIFIERS, DecisionRegressor, ScoreRegressor, Float32Regressor, OUT, REVISION, ROOT, driver as ovr_driver, lit, matrix, parse


class Float32DecisionRegressor(DecisionRegressor, Float32Regressor):
    pass


class ProbabilityRegressor(ScoreRegressor):
    def predict_proba(self, X):
        model = self.model_
        old = model.shift
        model.shift = self.shift + 10.0
        try:
            return model._proba(X)
        finally:
            model.shift = old


COMPONENTS = {**CLASSIFIERS, "DecisionRegressor": DecisionRegressor,
              "Float32DecisionRegressor": Float32DecisionRegressor,
              "ProbabilityRegressor": ProbabilityRegressor}

def cases():
    rng = np.random.default_rng(20261004)
    result = []
    for n_classes in (1, 2, 3, 5):
        for component in COMPONENTS:
            for code_size, seed in ((1.5, 0), (2.0, 1), (1.0 if n_classes == 1 else 0.75, 42)):
                n = 6 * n_classes
                X = rng.normal(size=(n, 3))
                y = np.tile(np.arange(n_classes), 6)
                rng.shuffle(y)
                X += y[:, None] * 0.3
                result.append(dict(name=f'{component}-k{n_classes}-size{code_size}-seed{seed}',
                                   component=component, code_size=code_size, seed=seed,
                                   label='int64', shift=0.0, X=X, y=y * 3 - 7,
                                   Xt=rng.normal(size=(7, 3)), jobs=1, column=False))
    # Exact duplicate words and equidistant words on symmetric training data.
    for component in COMPONENTS:
        result.append(dict(name=f'{component}-duplicate-words', component=component,
                           code_size=0.34, seed=0, label='int64', shift=0.0, jobs=2,
                           X=np.array([[0., 0.], [0., 0.], [0., 0.]]),
                           y=np.array([7, -2, 3]), Xt=np.array([[0., 0.], [1., 1.]]), column=False))
    for component in COMPONENTS:
        source = next(c for c in result if c['name'].endswith('duplicate-words')).copy()
        source.update(name=f'{component}-equidistant-words', component=component,
                      code_size=2.0, seed=17)
        result.append(source)
    for label in ('float32', 'float64'):
        for component in COMPONENTS:
            source = result[24].copy()
            source.update(name=f'{component}-{label}-column-maxseed', label=label,
                          component=component, seed=2**32-1, column=True, jobs=2)
            source['y'] = source['y'].astype(label).reshape(-1, 1)
            result.append(source)
    # More than 312 uniform draws: crosses an MT19937 twist boundary.
    source = result[27].copy()
    source.update(name='mt-twist-boundary', code_size=16.0, seed=17)
    result.append(source)
    # None consumes consecutive portions of the process stream, with explicit reset.
    for i in range(2):
        source = result[18].copy()
        source.update(name=f'global-stream-{i}', seed=None, reset=i == 0, jobs=2)
        result.append(source)
    return result


def reference(case):
    if case.get('reset'):
        np.random.seed(123)
    model = OutputCodeClassifier(COMPONENTS[case['component']](shift=case['shift']),
                                 code_size=case['code_size'], random_state=case['seed'],
                                 n_jobs=case['jobs']).fit(case['X'], case['y'])
    scores = np.array([_predict_binary(e, case['Xt']) for e in model.estimators_]).T
    return dict(classes=model.classes_.tolist(), book=model.code_book_.tolist(),
                n_estimators=len(model.estimators_),
                constants=[int(hasattr(e, 'y_')) for e in model.estimators_],
                predict=model.predict(case['Xt']).tolist(), confidence=scores.tolist(),
                score=model.score(case['X'], case['y']),
                wscore=model.score(case['X'], case['y'], sample_weight=np.arange(1, len(case['y']) + 1.)))


def driver(all_cases):
    header = ovr_driver([]).split('def main() raises:')[0].replace(
        'from moxie.multiclass import OneVsRestClassifier',
        'from moxie.multiclass import OutputCodeClassifier\n'
        'from moxie._multiclass_binary import predict_binary\n'
        'from moxie._random import seed_global_random_state')
    header += 'from ovr_fixtures import Float32DecisionRegressor, ProbabilityRegressor\n'
    body = []
    for idx, c in enumerate(all_cases):
        label = 'DType.' + c['label']
        body += [f'def case_{idx}() raises:',
                 f'    var X = mat({matrix(c["X"])})',
                 f'    var Xt = mat({matrix(c["Xt"])})',
                 f'    var flat = vec[{label}]([{", ".join(lit(v) for v in c["y"].ravel())}])']
        if c['column']:
            body += [f'    var y = nm.zeros[{label}](Shape(flat.size, 1))',
                     '    for i in range(flat.size):', '        y[Item(i, 0)] = flat[Item(i)]']
        else:
            body += ['    var y = flat.copy()']
        if c.get('reset'):
            body += ['    seed_global_random_state(123)']
        seed = 'None' if c['seed'] is None else str(c['seed'])
        body += [f'    var m = OutputCodeClassifier[{c["component"]}, label={label}]('
                 f'{c["component"]}(), code_size={c["code_size"]}, random_state={seed}, n_jobs={c["jobs"]})',
                 '    m.fit(X, y)',
                 '    var confidence = nm.zeros[DType.float64](Shape(Xt.shape[0], len(m.estimators_)))',
                 '    var constants = nm.zeros[DType.int64](Shape(len(m.estimators_)))',
                 '    for k in range(len(m.estimators_)):',
                 '        ref e = m.estimators_[k]',
                 '        if e.is_constant():',
                 '            constants[Item(k)] = 1',
                 '            for i in range(Xt.shape[0]):',
                 '                confidence[Item(i, k)] = Float64(e.constant())',
                 '        else:',
                 '            var col = predict_binary(e.estimator(), Xt, Xt.shape[0])',
                 '            for i in range(Xt.shape[0]):',
                 '                confidence[Item(i, k)] = col[Item(i)]',
                 '    var w = nm.zeros[DType.float64](Shape(flat.size))',
                 '    for i in range(w.size):', '        w[Item(i)] = Float64(i + 1)',
                 f'    print(\'{{"case": {idx},\')',
                 '    emit("classes", m.classes_)', '    emit("book", m.code_book_)',
                 '    emit("constants", constants)', '    emit("confidence", confidence)',
                 '    emit("predict", m.predict(Xt))',
                 '    print(\'"n_estimators":\', len(m.estimators_), ",")',
                 '    print(\'"score":\', m.score(X, y), ",")',
                 '    print(\'"wscore":\', m.score(X, y, w.copy()), ",")',
                 '    print(\'"end": 0}\')', '']
    body += ['def main() raises:'] + [f'    case_{i}()' for i in range(len(all_cases))]
    return header + '\n'.join(body) + '\n'


def main():
    from sklearn.utils._param_validation import InvalidParameterError
    try:
        OutputCodeClassifier(ScoreRegressor()).fit([[0.], [1.]], [0, 1])
    except InvalidParameterError:
        pass
    else:
        raise AssertionError("plain regressors must fail parameter validation")
    class ColumnDecisionRegressor(DecisionRegressor):
        def predict(self, X):
            return super().predict(X).reshape(-1, 1)
    model = OutputCodeClassifier(ColumnDecisionRegressor(), random_state=17, code_size=2)
    X = np.array([[0.], [1.], [2.]])
    model.fit(X, [0, 1, 2])
    try:
        model.predict(X)
    except ValueError:
        pass
    else:
        raise AssertionError("column-shaped regressor predictions must be rejected")
    all_cases = cases()
    expected = [reference(c) for c in all_cases]
    source, binary = OUT / 'output_code_driver.mojo', OUT / 'output_code_driver'
    source.write_text(driver(all_cases))
    command = ['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT / 'builds/deps',
               '-I', ROOT / 'tests/multiclass', source, '-o', binary]
    subprocess.run(command, check=True)
    got = parse(subprocess.run([binary], check=True, capture_output=True, text=True).stdout)
    assert len(got) == len(expected)
    failures = []
    worst = 0.
    for case, ref, out in zip(all_cases, expected, got):
        checks = {key: np.array_equal(ref[key], out[key]) for key in
                  ('classes', 'book', 'constants', 'predict', 'n_estimators', 'score')}
        checks['wscore'] = math.isclose(ref['wscore'], out['wscore'], rel_tol=1e-15, abs_tol=1e-15)
        tolerance = 2e-6 if case['component'] == 'Float32DecisionRegressor' else (5e-11 if case['component'] == 'ProbaClassifier' else 1e-12)
        a, b = np.asarray(ref['confidence']), np.asarray(out['confidence'])
        checks['confidence'] = a.shape == b.shape and np.allclose(a, b, rtol=tolerance, atol=1e-14)
        if a.shape == b.shape:
            worst = max(worst, float(np.max(np.abs(a-b))))
        if not all(checks.values()):
            failures.append((case['name'], [k for k,v in checks.items() if not v]))
    report = dict(revision=REVISION, sklearn=sklearn.__version__, cases=len(all_cases),
                  command=[str(x) for x in command], seeds=[0, 1, 42, 17, 2**32-1, 'global:123'],
                  max_confidence_absolute_error=worst, failures=failures)
    (OUT / 'output-code-parity-report.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'output-code-expected.json').write_text(json.dumps(expected, indent=2) + '\n')
    (OUT / 'output-code-actual.json').write_text(json.dumps(got, indent=2) + '\n')
    assert not failures, failures
    print(f'PASS: {len(all_cases)} OutputCodeClassifier cases match scikit-learn {sklearn.__version__}')


if __name__ == '__main__':
    main()
