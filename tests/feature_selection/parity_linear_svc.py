"""Compare dense native LinearSVC selectors with the pinned sklearn sources."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
OUTPUT = ROOT / 'builds/linear-svc-selection-parity'
sys.path.insert(0, str(ROOT / 'tools'))
from compiler_limits import compiler_command


def reference_module(package, module):
    source = REFERENCE / 'sklearn' / package / f'{module}.py'
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
    spec = importlib.util.spec_from_file_location(f'sklearn.{package}._moxie_reference{module}', source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HEADER = """\
import numojo as nm
from moxie.base import clone
from moxie.feature_selection import DenseSupervisedCV, FeatureCount, LinearSVCImportance, RFE, RFECV, SelectFromModel, SequentialCV
from moxie.svm import LinearSVC
from std.testing import assert_equal, assert_false, assert_raises

def dump[dtype: DType](values: nm.NDArray[dtype]) raises:
    print(values.ndim)
    for axis in range(values.ndim):
        print(values.shape[axis])
    for i in range(values.size):
        print(values.unsafe_get(i))
"""


def fixture(selector_name, dtype, label, penalty, SVC, SelectFromModel, rfe):
    multiclass = label == 'String'
    labels = ['é', 'alpha', 'z'] if multiclass else [2**53 + 1, 2**53]
    rng = np.random.RandomState(124)
    groups = rng.permutation(np.tile(np.arange(len(labels)), 6 if multiclass else 9))
    X = rng.normal(scale=.4, size=(len(groups), 4))
    X[:, 2] += 2 * groups
    X[:, 0] += 1.4 * (groups == 1)
    X = X.astype(dtype)
    y = np.array(labels)[groups]
    weights = np.linspace(.75, 1.75, len(y)).astype(dtype)
    model = SVC(C=.7, penalty=penalty, dual=False, random_state=11, tol=1e-9, max_iter=5000)
    getter = 'coef_' if multiclass else 'auto'
    if selector_name == 'SelectFromModel':
        selector = SelectFromModel(model, importance_getter=getter)
    elif selector_name == 'RFE':
        selector = rfe.RFE(model, n_features_to_select=2, importance_getter=getter)
    else:
        selector = rfe.RFECV(model, cv=3, importance_getter=getter)
    weighted = selector_name != 'RFECV'
    selector.fit(X, y, **({'sample_weight': weights} if weighted else {}))

    lines = [HEADER, f'comptime E = LinearSVCImportance[{label}, DType.{dtype}]']
    adapter = 'DenseSupervisedCV[E, E.dtype]' if selector_name == 'RFECV' else 'E'
    lines += [f'comptime S = {selector_name}[{adapter}]']
    lines += ['def main() raises:',
              f'    var model = LinearSVC[{label}](C=0.7, penalty="{penalty}", dual=False, random_state=11, tol=1e-9, max_iter=5000)']
    numbers = []
    comparisons = 0

    def add(line):
        lines.append('    ' + line)

    def literal(value):
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (bool, np.bool_)):
            return 'True' if value else 'False'
        return str(value)

    def exact(expression, expected):
        nonlocal comparisons
        add(f'assert_equal({expression}, {literal(expected)})')
        comparisons += 1

    def exact_list(name, expected, *, shaped=False):
        exact(f'len({name})' if not shaped else f'{name}.size', len(expected))
        for i, value in enumerate(expected):
            index = f'nm.Item({i})' if shaped else str(i)
            exact(f'{name}[{index}]', value)

    def dump(expression, expected, *, dtype=None):
        values = np.asarray(expected, dtype=dtype)
        add(f'dump({expression})')
        numbers.extend([values.ndim, *values.shape, *values.flat])

    def input_array(name, values):
        # Exercise logical row/column order with negative strides and a nonzero offset.
        shape = ', '.join(map(str, values.shape))
        strides = '-4, -1' if values.ndim == 2 else '-1'
        add(f'var {name}_backing = nm.zeros[DType.{dtype}](nm.Shape({values.size + 1}))')
        add(f'var {name} = {name}_backing.view_with_layout(nm.Shape({shape}), nm.NDArrayStrides({strides}), {values.size})')
        for index in np.ndindex(values.shape):
            item = ', '.join(map(str, index))
            add(f'{name}[nm.Item({item})] = {float(values[index])!r}')

    input_array('X', X)
    add(f'var y: List[{label}] = [{", ".join(map(literal, y.tolist()))}]')
    if weighted:
        input_array('weights', weights)
    if selector_name == 'RFECV':
        add('var adapter = DenseSupervisedCV[E, E.dtype]()')
        add(f'var splits = SequentialCV(3).splits({len(y)}, adapter.cv_labels(y), adapter.is_classifier())')
        for fold, (train, test) in enumerate(StratifiedKFold(3).split(X, y)):
            exact_list(f'splits[{fold}].train', train, shaped=True)
            exact_list(f'splits[{fold}].test', test, shaped=True)
    options = {'SelectFromModel': '', 'RFE': ', n_features_to_select=FeatureCount(2)',
               'RFECV': ', cv=SequentialCV(3)'}
    add(f'var selector = S(E(model^), importance_getter="{getter}"{options[selector_name]})')
    add('with assert_raises(contains="NotFittedError"):')
    add('    _ = selector.transform(X)')
    target = 'y.copy()' if selector_name == 'SelectFromModel' else 'y'
    add(f'_ = selector.fit(X, {target}{", sample_weight=weights" if weighted else ""})')
    add('assert_false(selector.estimator.__sklearn_is_fitted__())')
    add('assert_false(clone(selector).__sklearn_is_fitted__())')
    add('var mask = selector.get_support()[List[Bool]].copy()')
    exact_list('mask', selector.get_support())
    dump('selector.transform(X)', selector.transform(X))
    add('var fitted = selector.estimator_()')
    add('var classes = fitted.classes_()')
    exact_list('classes', selector.estimator_.classes_.tolist())
    dump('fitted.importances("coef_")', selector.estimator_.coef_, dtype=dtype)
    if selector_name == 'SelectFromModel':
        dump('fitted.decision_function(X)', selector.estimator_.decision_function(X))
        add('print(selector.threshold_())')
        numbers.append(selector.threshold_)
    else:
        add('var ranking = selector.ranking_()')
        exact_list('ranking', selector.ranking_)
        add('var prediction = selector.predict(X)')
        exact_list('prediction', selector.predict(X).tolist())
        dump('selector.decision_function(X)', selector.decision_function(X))
        weight_arg = ', weights.copy()' if weighted else ''
        add(f'print(selector.score(X, y{weight_arg}))')
        numbers.append(selector.score(X, y, **({'sample_weight': weights} if weighted else {})))
    if selector_name == 'RFECV':
        add('var results = selector.cv_results_()')
        for key, column in selector.cv_results_.items():
            name = f'result_{key}'
            if key == 'n_features':
                add(f'var {name} = results["{key}"][List[Int]].copy()')
                exact_list(name, column)
            elif key.endswith(('_support', '_ranking')):
                typ = 'Bool' if key.endswith('_support') else 'Int'
                add(f'var {name} = results["{key}"][List[List[{typ}]]].copy()')
                exact(f'len({name})', len(column))
                for row, values in enumerate(column):
                    exact_list(f'{name}[{row}]', values)
            else:
                dump(f'results["{key}"][nm.NDArray[DType.float64]]', column)
    return '\n'.join(lines) + '\n', numbers, comparisons


def main():
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    svc = reference_module('svm', '_classes').LinearSVC
    model = reference_module('feature_selection', '_from_model').SelectFromModel
    rfe = reference_module('feature_selection', '_rfe')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assembly = ROOT / 'builds/deps'
    count = 0
    for selector in ('RFECV', 'RFE', 'SelectFromModel'):
        for dtype, label in [('float32', 'String'), ('float64', 'Int64')]:
            penalty = 'l1' if selector == 'SelectFromModel' and label == 'Int64' else 'l2'
            code, expected, comparisons = fixture(selector, dtype, label, penalty, svc, model, rfe)
            source = OUTPUT / f'{selector}-{dtype}.mojo'
            source.write_text(code)
            binary = source.with_suffix('')
            with source.with_suffix('.log').open('w') as log:
                try:
                    subprocess.run(compiler_command(['mojo', 'build', '-O3', '-I', assembly, source, '-o', binary]),
                                   check=True, stdout=log, stderr=log)
                except subprocess.CalledProcessError:
                    print(f'Compiler log: {log.name}', file=sys.stderr)
                    raise
            actual = np.array(subprocess.check_output([binary], text=True).split(), dtype=float)
            # LIBLINEAR termination and Float32 importance storage can round
            # coefficients and scores. Masks, class labels and folds are exact.
            tolerance = 3e-6 if dtype == 'float32' else 2e-7
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            count += comparisons + len(expected)
            print(f'PASS {selector}/{dtype}/{label}/{penalty} ({comparisons + len(expected)} comparisons)', flush=True)
    print(f'PASS LinearSVC selector parity ({count} comparisons)')


if __name__ == '__main__':
    main()
