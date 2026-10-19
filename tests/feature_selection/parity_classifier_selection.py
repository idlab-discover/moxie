"""Small production-classifier selector fixtures against the local sklearn classes.

Each selector/label case is compiled independently through tools.check, preserving
its serial compiler lock, ten-minute timeout and hard 8 GiB address-space cap.
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
OUTPUT = ROOT / 'builds/classifier-selection-parity'


def oracle():
    import numpy as np
    import sklearn
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    assert sklearn.__version__ in ('1.9.1', '1.10.dev0')
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION

    def load(package, name):
        source = REFERENCE / 'sklearn' / package / (name + '.py')
        assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
        spec = importlib.util.spec_from_file_location(f'sklearn.{package}._selection_reference{name}', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    tree = load('tree', '_classes').DecisionTreeClassifier
    rfe = load('feature_selection', '_rfe')
    model = load('feature_selection', '_from_model').SelectFromModel
    sequential = load('feature_selection', '_sequential').SequentialFeatureSelector
    cases = []
    for dtype, strings, classes in [('float32', True, 2), ('float64', True, 3), ('float64', False, 2), ('float32', False, 3)]:
        rng = np.random.RandomState(714)
        x = rng.normal(size=(24, 4)).astype(dtype)
        groups = np.argsort(np.argsort(2*x[:, 2] - x[:, 0])) * classes // len(x)
        labels = np.array(['é', 'alpha', 'z']) if strings else np.array([2**53+1, 2**53, -2**63], dtype=np.int64)
        y = labels[groups]
        x2 = x[:, [2, 0, 1]].copy()
        y2 = labels[(groups + 1) % 2]  # refit drops a class for multiclass inputs
        estimator = tree(max_depth=3, min_samples_leaf=2, random_state=42)
        for name in ['SelectFromModel', 'RFE', 'RFECV', 'SequentialFeatureSelector']:
            selector = {'SelectFromModel': lambda: model(estimator),
                        'RFE': lambda: rfe.RFE(estimator, n_features_to_select=2),
                        'RFECV': lambda: rfe.RFECV(estimator, cv=3),
                        'SequentialFeatureSelector': lambda: sequential(estimator, n_features_to_select=2, cv=3, direction='backward' if classes == 3 else 'forward')}[name]()
            case = dict(dtype=dtype, strings=strings, classes=classes, selector=name, x=x.tolist(), y=y.tolist(), x2=x2.tolist(), y2=y2.tolist(), phases=[])
            splits = list(StratifiedKFold(3).split(x, y))
            case['splits'] = [[a.tolist(), b.tolist()] for a, b in splits]
            case['scores'] = cross_val_score(estimator, x, y, cv=splits).tolist()
            for X, Y in [(x, y), (x2, y2)]:
                selector.fit(X, Y)
                phase = dict(mask=selector.get_support().tolist())
                assert not hasattr(clone(selector), 'n_features_in_')
                if name in ['RFE', 'RFECV']:
                    phase.update(ranking=selector.ranking_.tolist(), prediction=selector.predict(X).tolist(), classes=selector.classes_.tolist(), probability=selector.predict_proba(X).tolist(), log_probability=selector.predict_log_proba(X).tolist(), score=selector.score(X, Y))
                    assert not hasattr(selector, 'decision_function')
                elif name == 'SelectFromModel':
                    phase.update(classes=selector.estimator_.classes_.tolist(), importances=selector.estimator_.feature_importances_.tolist(), threshold=selector.threshold_)
                    assert not hasattr(selector, 'predict')
                else:
                    assert not hasattr(selector, 'predict')
                if name == 'RFECV':
                    phase['cv_results'] = {k: v.tolist() for k, v in selector.cv_results_.items()}
                case['phases'].append(phase)
            cases.append(case)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'oracle.json').write_text(json.dumps(dict(revision=REVISION, infrastructure=sklearn.__version__, cases=cases), indent=2, ensure_ascii=False))


def main(start_case=0):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools'))
    import feature_selection_checks as check
    python = os.environ.get('MOXIE_REFERENCE_PYTHON', str(ROOT.parent / 'moxie/builds/tree/oracle-env/bin/python'))
    subprocess.run([python, str(Path(__file__).resolve()), '--oracle'], check=True)
    cases = json.loads((OUTPUT / 'oracle.json').read_text())['cases']
    if not 0 <= start_case < len(cases):
        raise ValueError('start-case is outside the fixture range')
    for number, case in enumerate(cases):
        if number < start_case:
            continue
        dt, strings, name = case['dtype'], case['strings'], case['selector']
        lines = ['import numojo as nm', 'from moxie._labels import StringTarget',
                 'from moxie.tree import DecisionTreeClassifier',
                 'from moxie.feature_selection import DecisionTreeClassifierImportance, DenseSupervisedCV, SelectFromModel, RFE, RFECV, SequentialFeatureSelector, SequentialCV, SequentialCount, FeatureCount, RFECVResults',
                 'from moxie.base import clone, Predictor, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction, SupervisedEstimator',
                 'from moxie.model_selection._validation import cross_val_score',
                 'from std.testing import assert_equal, assert_true, assert_false, assert_raises',
                 f'comptime E = DecisionTreeClassifierImportance[DType.{dt}, DType.int64, {strings}]',
                 f'comptime A = DenseSupervisedCV[E, DType.{dt}]',
                 f'comptime S = {name}[{"A" if name in ["RFECV", "SequentialFeatureSelector"] else "E"}]',
                 '@no_inline',
                 ('def fit_selector(mut selector: S, X: E.Input, y: E.Target) raises:' if name == 'SelectFromModel' else 'def fit_selector[T: SupervisedEstimator](mut selector: T, X: T.Input, y: T.Target) raises:'),
                 f'    _ = selector.fit(X, y{".copy()" if name == "SelectFromModel" else ""})',
                 'def main() raises:',
                 f'    comptime assert {"" if name in ["RFE", "RFECV"] else "not "}conforms_to(S, Predictor)',
                 f'    comptime assert {"" if name in ["RFE", "RFECV"] else "not "}conforms_to(S, ProbabilityPredictor)',
                 '    comptime assert not conforms_to(S, DecisionFunction)',
                 f'    var estimator = E(DecisionTreeClassifier[DType.int64, {strings}](max_depth=3, min_samples_leaf=2, random_state=42))']
        expected = []
        phase_checks = []

        def array(var, values):
            a = np.asarray(values)
            # Reverse rows and columns via genuine views; fixtures remain tiny.
            shape = ', '.join(map(str, a.shape))
            lines.append(f'    var {var}_backing = nm.zeros[DType.{dt}](nm.Shape({a.size + 2}))')
            lines.append(f'    var {var} = {var}_backing.view_with_layout(nm.Shape({shape}), nm.NDArrayStrides(-{a.shape[1]}, -1), {a.size})')
            for i, row in enumerate(a):
                for j, value in enumerate(row):
                    lines.append(f'    {var}[nm.Item({i}, {j})] = {float(value)!r}')

        def target(var, values):
            if strings:
                lines.append(f'    var {var} = StringTarget({json.dumps(values, ensure_ascii=False)})')
            else:
                lines.append(f'    var {var}_backing = nm.zeros[DType.int64](nm.Shape({2*len(values)+1}))')
                lines.append(f'    var {var} = {var}_backing.view_with_layout(nm.Shape({len(values)}), nm.NDArrayStrides(-2), {2*len(values)-1})')
                for i, value in enumerate(values):
                    literal = '-9223372036854775807 - 1' if value == -2**63 else str(value)
                    lines.append(f'    {var}[nm.Item({i})] = Int64({literal})')

        def exact(expr, values):
            for i, v in enumerate(values):
                literal = json.dumps(v, ensure_ascii=False) if isinstance(v, str) else ('True' if v is True else 'False' if v is False else (str(v) if v != -2**63 else "Int64(-9223372036854775807 - 1)"))
                lines.append(f'    assert_equal({expr.format(i=i)}, {literal})')

        def numbers(expr, values):
            for i, v in enumerate(np.asarray(values).flat):
                lines.append(f'    print({expr.format(i=i)})')
                expected.append(float(v))

        array('x', case['x']); target('y', case['y'])
        lines += ['    var adapter = A()', '    var cv = SequentialCV(3)', '    var splits = cv.splits(24, adapter.cv_labels(y), adapter.is_classifier())']
        for fold, (train, test) in enumerate(case['splits']):
            exact(f'splits[{fold}].train[nm.Item({{i}})]', train)
            exact(f'splits[{fold}].test[nm.Item({{i}})]', test)
        lines.append('    var scores = cross_val_score[A](estimator, x, y, adapter, adapter.default_scorer(), splits, None, None)')
        numbers('scores.unsafe_get({i})', case['scores'])
        args = {'SelectFromModel': '', 'RFE': ', n_features_to_select=FeatureCount(2)', 'RFECV': ', cv=SequentialCV(3)', 'SequentialFeatureSelector': f', cv=SequentialCV(3), n_features_to_select=SequentialCount(2), direction="{"backward" if case["classes"] == 3 else "forward"}"'}[name]
        lines.append(f'    var selector = S(estimator.copy(){args})')
        lines += ['    with assert_raises():', '        _ = selector.transform(x)']
        if name in ['RFE', 'RFECV']:
            lines += ['    comptime assert conforms_to(S, LogProbabilityPredictor)', '    with assert_raises():', '        _ = selector.predict_proba(x)']
        for phase_id, phase in enumerate(case['phases']):
            x, y = ('x', 'y') if phase_id == 0 else ('x2', 'y2')
            if phase_id:
                array(x, case[x]); target(y, case[y])
                lines += ['    var params = selector.get_params()', '    params.estimator.model.params.max_depth = 3', '    selector.set_params(params)']
            # SelectFromModel accepts an owned Optional target; use explicit copy.
            lines.append(f'    fit_selector(selector, {x}, {y})')
            lines.append(f'    var mask{phase_id} = selector.get_support()[List[Bool]].copy()')
            exact(f'mask{phase_id}[{{i}}]', phase['mask'])
            lines += [f'    var fresh{phase_id} = clone(selector)', f'    assert_false(fresh{phase_id}.__sklearn_is_fitted__())', f'    assert_false(fresh{phase_id}.estimator.__sklearn_is_fitted__())', '    assert_false(estimator.__sklearn_is_fitted__())']
            if name in ['RFE', 'RFECV']:
                lines += [f'    var ranks{phase_id} = selector.ranking_()', f'    var prediction{phase_id} = selector.predict({x})', f'    var classes{phase_id} = selector.classes_()', f'    var probability{phase_id} = selector.predict_proba({x})', f'    var log_probability{phase_id} = selector.predict_log_proba({x})']
                exact(f'ranks{phase_id}[{{i}}]', phase['ranking'])
                exact(f'prediction{phase_id}.get({{i}})' if strings else f'prediction{phase_id}[nm.Item({{i}})]', phase['prediction'])
                exact(f'classes{phase_id}[{{i}}]' if strings else f'classes{phase_id}[nm.Item({{i}})]', phase['classes'])
                numbers(f'probability{phase_id}.unsafe_get({{i}})', phase['probability'])
                numbers(f'log_probability{phase_id}.unsafe_get({{i}})', phase['log_probability'])
                numbers(f'selector.score({x}, {y})', [phase['score']])
            elif name == 'SelectFromModel':
                lines += [f'    var fitted{phase_id} = selector.estimator_()', f'    var classes{phase_id} = fitted{phase_id}.classes_()', f'    var importances{phase_id} = fitted{phase_id}.feature_importances()']
                exact(f'classes{phase_id}[{{i}}]' if strings else f'classes{phase_id}[nm.Item({{i}})]', phase['classes'])
                numbers(f'importances{phase_id}.unsafe_get({{i}})', phase['importances'])
                numbers('selector.threshold_()', [phase['threshold']])
            if name == 'RFECV':
                lines.append(f'    var results{phase_id} = selector.cv_results_()')
                check_start = len(lines)
                for key, values in phase['cv_results'].items():
                    column = f'result_{phase_id}_{key}'
                    if key == 'n_features':
                        lines.append(f'    var {column} = results{phase_id}["{key}"][List[Int]].copy()')
                        exact(f'{column}[{{i}}]', values)
                    elif key.endswith(('_support', '_ranking')):
                        typ = 'Bool' if key.endswith('_support') else 'Int'
                        lines.append(f'    var {column} = results{phase_id}["{key}"][List[List[{typ}]]].copy()')
                        for row, vals in enumerate(values):
                            exact(f'{column}[{row}][{{i}}]', vals)
                    else:
                        lines.append(f'    var {column} = results{phase_id}["{key}"][nm.NDArray[DType.float64]].copy()')
                        numbers(f'{column}.unsafe_get({{i}})', values)
            if name == 'RFECV':
                # Keep each fitted-state check out of main's compiler stack frame.
                phase_checks += ['@no_inline', f'def verify_phase_{phase_id}(results{phase_id}: RFECVResults) raises:', *lines[check_start:]]
                lines[check_start:] = [f'    verify_phase_{phase_id}(results{phase_id})']
        if phase_checks:
            main_start = lines.index('def main() raises:')
            lines[main_start:main_start] = phase_checks
        path = OUTPUT / f'case_{number}.mojo'
        path.write_text('\n'.join(lines) + '\n')
        check.compile_file(ROOT, path, f'classifier-parity-{number}')
        result = check.run(ROOT / f'builds/classifier-parity-{number}', capture=True, check=False)
        (OUTPUT / f'case_{number}.stdout').write_text(result.stdout)
        (OUTPUT / f'case_{number}.stderr').write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = np.array([float(line) for line in result.stdout.splitlines()])
        # Tree arithmetic uses float64, importances retain feature dtype.
        np.testing.assert_allclose(actual, expected, rtol=2e-6 if dt == 'float32' else 1e-12, atol=1e-12)
        print(f'PASS classifier selector {number}: {name}, {dt}, {"strings" if strings else "Int64"}, {case["classes"]} classes', flush=True)
    print(f'PASS {len(cases) - start_case} independent classifier selector fixtures (starting at {start_case})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--start-case', type=int, default=0)
    args = parser.parse_args()
    oracle() if args.oracle else main(args.start_case)
