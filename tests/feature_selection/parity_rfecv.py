"""Compare actual pinned sklearn RFECV and native real Lasso/ElasticNet paths."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
OUTPUT = ROOT / 'builds/rfecv-parity'


def oracle():
    import numpy as np
    import sklearn
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.model_selection import KFold
    from sklearn.metrics import get_scorer
    from scipy.linalg import hadamard
    assert sklearn.__version__ == '1.9.1'
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    source = REFERENCE / 'sklearn/feature_selection/_rfe.py'
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
    spec = importlib.util.spec_from_file_location('sklearn.feature_selection._moxie_reference_rfecv', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    rng = np.random.RandomState(905)
    for dtype in ['float32', 'float64']:
        for model in ['Lasso', 'ElasticNet']:
            for step, minimum in [(1, 1), (2, 1), (3, 2), (.4, 1), (.01, 2), (20, 1), (2, 5), (1, 9)]:
                x = np.tile(hadamard(8)[:, 1:6], (5, 1)).astype(dtype)
                y = (4*x[:, 0] - 2*x[:, 2]).astype(dtype)
                cases.append(dict(dtype=dtype, model=model, step=step, minimum=minimum, x=x, y=y, mode=0, weights=None, score_weights=None, reverse=False, custom=False))
            for mode in [1, 2, 4, 5, 6]:
                cases.append(dict(dtype=dtype, model=model, step=2, minimum=1, x=x.copy(), y=y.copy(), mode=mode, weights=None, score_weights=None, reverse=False, custom=False))
            for outputs, weighted in [(1, False), (2, False), (1, True), (2, True)]:
                x = rng.normal(size=(36, 5)).astype(dtype)
                y = (.3*x[:, 0] + 2*x[:, 1] + .4*x[:, 2] - 3*x[:, 3] + .2*x[:, 4] + .5*rng.normal(size=36)).astype(dtype)
                if outputs == 2:
                    y = np.column_stack((y, -y + .1*x[:, 0])).astype(dtype)
                weights = np.linspace(.5, 2, 36).astype(dtype) if weighted else None
                score_weights = np.linspace(3, .5, 36).astype(dtype) if weighted else None
                cases.append(dict(dtype=dtype, model=model, step=.4, minimum=1, x=x[::-1, ::-1], y=y[::-1], mode=0, weights=weights, score_weights=score_weights, reverse=True, custom=True))
    for case in cases:
        estimator = (Lasso if case['model'] == 'Lasso' else ElasticNet)(alpha=.08, fit_intercept=False, max_iter=5000, tol=1e-9)
        mode = case['mode']
        def score(est, x, y):
            if mode == 1: return 7.
            if mode == 2: return float(x.shape[1])
            if mode == 4: return float('nan') if x.shape[1] == 3 else float(x.shape[1])
            if mode == 5: return float('inf')
            if mode == 6: return -float('inf')
            return est.score(x, y)
        cv = list(KFold(3).split(case['x'])) if case['custom'] else None
        selector = module.RFECV(estimator, step=case['step'], min_features_to_select=case['minimum'], cv=cv, scoring=score if mode else None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if case['weights'] is None:
                selector.fit(case['x'], case['y'])
                final_score = selector.score(case['x'], case['y'])
            else:
                with sklearn.config_context(enable_metadata_routing=True):
                    estimator.set_fit_request(sample_weight=True)
                    selector.scoring = get_scorer('r2').set_score_request(sample_weight='test_weights')
                    selector.fit(case['x'], case['y'], sample_weight=case['weights'], test_weights=case['score_weights'])
                    final_score = selector.score(case['x'], case['y'], test_weights=case['score_weights'])
        case['support'] = selector.support_.astype(int).tolist()
        case['ranking'] = selector.ranking_.tolist()
        case['cv_results'] = {k: v.tolist() for k, v in selector.cv_results_.items()}
        case['predictions'] = selector.predict(case['x']).tolist()
        case['coef'] = selector.estimator_.coef_.tolist()
        case['wrapper_score'] = final_score
        # Native TypedScorer receives the final adapter and selected columns,
        # not the RFECV wrapper. Keep the unadapted score as explicit evidence.
        case['final_score'] = score(selector.estimator_, selector.transform(case['x']), case['y']) if mode else final_score
        case['splits'] = [[a.tolist(), b.tolist()] for a, b in cv] if case['custom'] else None
        for key in ['x', 'y', 'weights', 'score_weights']:
            if case[key] is not None:
                case[key] = case[key].tolist()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'oracle.json').write_text(json.dumps(dict(revision=REVISION, infrastructure=sklearn.__version__, cases=cases), indent=2))


def main():
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools'))
    import feature_selection_checks as check
    python = os.environ.get('MOXIE_REFERENCE_PYTHON', sys.executable)
    subprocess.run([python, str(Path(__file__).resolve()), '--oracle'], check=True)
    cases = json.loads((OUTPUT / 'oracle.json').read_text())['cases']
    source = ['import numojo as nm', 'from moxie.linear_model import Lasso, ElasticNet',
              'from moxie.feature_selection import RFECV, LassoImportance, ElasticNetImportance, FeatureCount, SequentialCV',
              'from moxie.model_selection._indices import IndexSplit',
              'from rfecv_support import TestCV, TestScore']
    expected, tolerances, labels = [], [], []
    for number, case in enumerate(cases):
        dt, model = case['dtype'], case['model']
        source.append(f'@no_inline\ndef case_{number}() raises:')
        def array(name, values, dtype=dt):
            a = np.asarray(values)
            source.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({", ".join(map(str,a.shape))}))')
            for i, value in enumerate(a.flat):
                source.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({repr(float(value)) if dtype != "int64" else int(value)}))')
        def expect(values, label, exact=False):
            flat = np.asarray(values).ravel().tolist()
            expected.extend(flat)
            tolerances.extend([0 if exact else (5e-5 if dt == 'float32' else 3e-8)] * len(flat))
            labels.extend([f'case {number} {dt} {model} {label}'] * len(flat))
        x, y = np.asarray(case['x']), np.asarray(case['y'])
        array('backing', x[::-1, ::-1] if case['reverse'] else x)
        array('y', y)
        if case['reverse']:
            source.append(f'    var x = backing.view_with_layout(nm.Shape({x.shape[0]}, {x.shape[1]}), nm.NDArrayStrides(-{x.shape[1]}, -1), {x.size-1})')
        else:
            source.append('    var x = backing.view()')
        source.append('    var cv: SequentialCV' if case['splits'] else '    var cv = SequentialCV(None)')
        if case['splits']:
            source.append('    var pairs = List[IndexSplit]()')
            for i, (train, test) in enumerate(case['splits']):
                array(f'train{i}', train, 'int64'); array(f'test{i}', test, 'int64')
                source.append(f'    pairs.append(IndexSplit(train{i}^, test{i}^))')
            source.append('    cv = SequentialCV(pairs)')
        step = f'Float64({case["step"]})' if isinstance(case['step'], float) else repr(case['step'])
        source += [f'    comptime E = {model}Importance[DType.{dt}]',
                   f'    var estimator = E({model}[DType.{dt}](alpha=0.08, fit_intercept=False, max_iter=5000, tol=1e-9))',
                   f'    var selector = RFECV[TestCV[E]](estimator.copy(), step=FeatureCount({step}), min_features_to_select={case["minimum"]}, cv=cv.copy(), scoring=TestScore[E]({case["mode"]}))']
        for key in ['weights', 'score_weights']:
            source.append(f'    var {key}: Optional[nm.NDArray[DType.{dt}]]' + (' = None' if case[key] is None else ''))
            if case[key] is not None:
                array(key+'_values', case[key])
                source.append(f'    {key} = {key}_values^')
        source += ['    selector.fit_named(x, y, List[String](), sample_weight=weights.copy(), score_metadata=score_weights.copy())',
                   '    var support = selector.support_()', '    var ranking = selector.ranking_()',
                   '    for j in range(len(support)):', '        print(Int(support[j]))',
                   '    for j in range(len(ranking)):', '        print(ranking[j])', '    print(selector.n_features_())',
                   '    var results = selector.cv_results_()']
        expect(case['support'], 'support', True)
        expect(case['ranking'], 'ranking', True)
        expect(sum(case['support']), 'count', True)
        for key, values in case['cv_results'].items():
            if key == 'n_features':
                source += [f'    for value in results["{key}"][List[Int]]:', '        print(value)']
                expect(values, key, True)
            elif key.endswith('_support') or key.endswith('_ranking'):
                typ = 'Bool' if key.endswith('_support') else 'Int'
                source += [f'    for row in results["{key}"][List[List[{typ}]]]:', '        for value in row:', '            print(Int(value))']
                expect(values, key, True)
            else:
                source += [f'    for j in range(results["{key}"][nm.NDArray[DType.float64]].size):', f'        print(results["{key}"][nm.NDArray[DType.float64]].unsafe_get(j))']
                expect(values, key)
        source += ['    var predictions = selector.predict(x)', '    for j in range(predictions.size):', '        print(predictions.unsafe_get(j))',
                   '    var fitted = selector.estimator_()', '    for j in range(fitted.model.coef_.size):', '        print(fitted.model.coef_.unsafe_get(j))',
                   '    print(selector.score(x, y, score_weights.copy()))']
        expect(case['predictions'], 'predictions'); expect(case['coef'], 'coefficients'); expect(case['final_score'], 'score')
    source.append('def main() raises:\n' + '\n'.join(f'    case_{i}()' for i in range(len(cases))))
    path = OUTPUT / 'main.mojo'
    path.write_text('\n'.join(source)+'\n')
    check.run('mojo', 'build', '-O3', '-I', ROOT/'builds/deps', '-I', ROOT/'tests/feature_selection', path, '-o', ROOT/'builds/rfecv-parity-run')
    result = check.run(ROOT / 'builds/rfecv-parity-run', capture=True)
    (OUTPUT / 'native.txt').write_text(result.stdout)
    actual = [float(v) for v in result.stdout.splitlines()]
    assert len(actual) == len(expected), (len(actual), len(expected))
    for got, want, tol, label in zip(actual, expected, tolerances, labels):
        if tol == 0:
            assert got == want, (label, got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=tol, atol=tol/10, equal_nan=True, err_msg=label)
    (OUTPUT / 'report.json').write_text(json.dumps(dict(cases=len(cases), comparisons=len(expected), revision=REVISION, seed=905,
        tolerances={'float32': {'rtol': 5e-5, 'atol': 5e-6}, 'float64': {'rtol': 3e-8, 'atol': 3e-9}},
        score_scope='Default score matches wrapper; custom score uses documented fitted-adapter/selected-input adaptation. Raw wrapper scores retained.',
        command='pixi run python tests/feature_selection/parity_rfecv.py', compiler=check.run('mojo', '--version', capture=True).stdout.strip()), indent=2))
    print(f'PASS RFECV parity: {len(cases)} real-model cases, {len(expected)} comparisons')


if __name__ == '__main__':
    oracle() if '--oracle' in sys.argv else main()
