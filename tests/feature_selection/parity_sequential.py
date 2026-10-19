"""Compare native SFS with the pinned sklearn class and real Lasso/ElasticNet.

The pinned selector runs on sklearn 1.9.1 infrastructure in the existing oracle
environment. No Python is involved in native selection or fitting.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
OUTPUT = ROOT / 'builds/sequential-parity'


def oracle():
    import numpy as np
    import sklearn
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.model_selection import KFold, cross_val_score
    from scipy.linalg import hadamard
    assert sklearn.__version__ == '1.9.1'
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    source = REFERENCE / 'sklearn/feature_selection/_sequential.py'
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
    spec = importlib.util.spec_from_file_location('sklearn.feature_selection._moxie_reference_sequential', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    rng = np.random.RandomState(713)
    for dtype in ['float32', 'float64']:
        for model in ['Lasso', 'ElasticNet']:
            for direction in ['forward', 'backward']:
                settings = [(1, None), (3, None), (.5, None), (1., None), ('auto', None), ('auto', .01)]
                settings += [(.01, None)] if direction == 'forward' else [('auto', -.01)]
                for count, tol in settings:
                    x = np.tile(hadamard(8)[:, 1:6], (5, 1)).astype(dtype)
                    y = (4 * x[:, 0] - 2 * x[:, 2]).astype(dtype)
                    cases.append(dict(dtype=dtype, model=model, direction=direction, count=count, tol=tol, x=x, y=y, custom=False, weights=None, reverse=False))
            # Nonorthogonal, strided data, multioutput and explicit fold lists.
            x = rng.normal(size=(24, 4)).astype(dtype)
            y = np.column_stack((2*x[:, 1] + .2*x[:, 2], -3*x[:, 3] + .1*x[:, 0])).astype(dtype)
            cases.append(dict(dtype=dtype, model=model, direction='backward', count=2, tol=None, x=x[::-1, ::-1], y=y[::-1], custom=True, weights=None, reverse=True))
            cases.append(dict(dtype=dtype, model=model, direction='forward', count=2, tol=None, x=x, y=y[:, 0], custom=True, weights=np.linspace(.5, 2, 24).astype(dtype), reverse=False))
    for case in cases:
        estimator = (Lasso if case['model'] == 'Lasso' else ElasticNet)(alpha=.08, fit_intercept=False, max_iter=5000, tol=1e-9)
        cv = list(KFold(3).split(case['x'])) if case['custom'] else 5
        selector = module.SequentialFeatureSelector(estimator, n_features_to_select=case['count'], direction=case['direction'], tol=case['tol'], cv=cv)
        if case['weights'] is None:
            selector.fit(case['x'], case['y'])
        else:
            with sklearn.config_context(enable_metadata_routing=True):
                estimator.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
                selector.fit(case['x'], case['y'], sample_weight=case['weights'])
        case['support'] = selector.get_support().astype(int).tolist()
        case['cv_scores'] = []
        if any(case['support']):
            if case['weights'] is None:
                case['cv_scores'] = cross_val_score(estimator, selector.transform(case['x']), case['y'], cv=cv).tolist()
            else:
                with sklearn.config_context(enable_metadata_routing=True):
                    case['cv_scores'] = cross_val_score(estimator, selector.transform(case['x']), case['y'], cv=cv, params=dict(sample_weight=case['weights'])).tolist()
        case['splits'] = [[a.tolist(), b.tolist()] for a, b in cv] if case['custom'] else None
        for key in ['x', 'y', 'weights']:
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
              'from moxie.feature_selection import SequentialFeatureSelector, DenseSupervisedCV, SequentialCount, SequentialCV',
              'from moxie.model_selection._indices import IndexSplit',
              'from moxie.model_selection._validation import cross_val_score']
    expected, tolerances = [], []
    for number, case in enumerate(cases):
        dt, model = case['dtype'], case['model']
        source.append(f'@no_inline\ndef case_{number}() raises:')
        def array(name, values, dtype=dt):
            a = np.asarray(values)
            source.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({", ".join(map(str,a.shape))}))')
            for i, value in enumerate(a.flat):
                source.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({repr(float(value)) if dtype != "int64" else int(value)}))')
        x, y = np.asarray(case['x']), np.asarray(case['y'])
        array('backing', x[::-1, ::-1] if case['reverse'] else x)
        array('y', y)
        if case['reverse']:
            source.append(f'    var x = backing.view_with_layout(nm.Shape({x.shape[0]}, {x.shape[1]}), nm.NDArrayStrides(-{x.shape[1]}, -1), {x.size-1})')
        else:
            source.append('    var x = backing.view()')
        source.append('    var cv = SequentialCV()')
        if case['splits']:
            source.append('    var pairs = List[IndexSplit]()')
            for i, (train, test) in enumerate(case['splits']):
                array(f'train{i}', train, 'int64'); array(f'test{i}', test, 'int64')
                source.append(f'    pairs.append(IndexSplit(train{i}^, test{i}^))')
            source.append('    cv = SequentialCV(pairs)')
        count = 'String("auto")' if case['count'] == 'auto' else (f'Float64({case["count"]})' if isinstance(case['count'], float) else repr(case['count']))
        tol = 'None' if case['tol'] is None else repr(case['tol'])
        source += [f'    comptime Adapter = DenseSupervisedCV[{model}[DType.{dt}], DType.{dt}]',
                   f'    var estimator = {model}[DType.{dt}](alpha=0.08, fit_intercept=False, max_iter=5000, tol=1e-9)',
                   f'    var selector = SequentialFeatureSelector[Adapter](estimator.copy(), n_features_to_select=SequentialCount({count}), direction="{case["direction"]}", tol={tol}, cv=cv.copy())']
        source.append(f'    var weights: Optional[nm.NDArray[DType.{dt}]] = None')
        if case['weights'] is not None:
            array('weight_values', case['weights'])
            source.append('    weights = weight_values^')
            source.append('    selector.fit_named(x, y, List[String](), fit_metadata=weights.copy(), score_metadata=weights.copy())')
        else:
            source.append('    selector.fit(x, y)')
        source += ['    var support = selector.support_()', f'    for j in range({x.shape[1]}):', '        print(Int(support[j]))', '    print(selector.n_features_to_select_())']
        expected += case['support'] + [sum(case['support'])]
        tolerances += [0] * (len(case['support']) + 1)
        if case['cv_scores']:
            source += ['    var chosen = selector.transform(x)',
                       '    var adapter = Adapter()',
                       '    var splits = cv.splits(x.shape[0], None, False)',
                       '    var scores = cross_val_score[Adapter](estimator, chosen, y, adapter, adapter.default_scorer(), splits, weights, weights)',
                       '    for j in range(scores.shape[0]):', '        print(scores.unsafe_get(j))']
            expected += case['cv_scores']
            tolerances += [5e-5 if dt == 'float32' else 3e-8] * len(case['cv_scores'])
    source.append('def main() raises:\n' + '\n'.join(f'    case_{i}()' for i in range(len(cases))))
    path = OUTPUT / 'main.mojo'
    path.write_text('\n'.join(source)+'\n')
    check.compile_file(ROOT, path, 'sequential-parity-run')
    result = check.run(ROOT / 'builds/sequential-parity-run', capture=True)
    (OUTPUT / 'native.txt').write_text(result.stdout)
    actual = [float(v) for v in result.stdout.splitlines()]
    assert len(actual) == len(expected), (len(actual), len(expected))
    for index, (got, want, tol) in enumerate(zip(actual, expected, tolerances)):
        if tol == 0:
            assert got == want, (index, got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=tol, atol=tol/10, err_msg=str(index))
    (OUTPUT / 'report.json').write_text(json.dumps(dict(cases=len(cases), comparisons=len(expected), revision=REVISION), indent=2))
    print(f'PASS sequential parity: {len(cases)} real-model cases, {len(expected)} comparisons')


if __name__ == '__main__':
    oracle() if '--oracle' in sys.argv else main()
