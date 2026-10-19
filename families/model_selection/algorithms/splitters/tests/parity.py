"""Execute pinned sklearn class bodies, then compare native integer folds exactly.

Only metadata and dense input validation are supplied by the harness. The actual
splitter classes and inherited algorithms are compiled from the pinned source.
A partial implementation never passes the full public-export acceptance gate.
"""
import ast
from abc import ABCMeta, abstractmethod
import hashlib
from itertools import combinations, chain
import json
import numbers
from math import ceil, floor
from collections import defaultdict
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings

import numpy as np
from collections.abc import Sequence
from scipy.sparse import issparse
from scipy.special import comb

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT.parents[4] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
REQUIRED = '''BaseCrossValidator BaseShuffleSplit KFold StratifiedKFold GroupKFold
StratifiedGroupKFold TimeSeriesSplit LeaveOneOut LeavePOut LeaveOneGroupOut
LeavePGroupsOut ShuffleSplit GroupShuffleSplit StratifiedShuffleSplit RepeatedKFold
RepeatedStratifiedKFold PredefinedSplit check_cv train_test_split'''.split()


def indexable(*arrays):
    sizes = [len(x) for x in arrays if x is not None]
    if sizes and any(n != sizes[0] for n in sizes):
        raise ValueError('Inconsistent lengths')
    return arrays


def check_array(x, **kwargs):
    x = np.array(x, copy=True)
    if not x.size:
        raise ValueError('Empty array')
    return x


def main():
    assert subprocess.check_output(['git', '-C', str(ORACLE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    source = ORACLE / 'sklearn/model_selection/_split.py'
    assert not subprocess.check_output(['git', '-C', str(ORACLE), 'diff', 'HEAD', '--', str(source)], text=True)
    classes = {'_UnsupportedGroupCVMixin', 'GroupsConsumerMixin', 'BaseCrossValidator',
               '_BaseKFold', 'KFold', 'TimeSeriesSplit', 'LeaveOneOut', 'LeavePOut',
               'LeaveOneGroupOut', 'LeavePGroupsOut', 'PredefinedSplit', 'StratifiedKFold', 'GroupKFold', 'BaseShuffleSplit', 'ShuffleSplit', 'GroupShuffleSplit', '_RepeatedSplits', 'RepeatedKFold', 'RepeatedStratifiedKFold', 'StratifiedGroupKFold', 'StratifiedShuffleSplit'}
    namespace = dict(np=np, numbers=numbers, warnings=warnings, ABCMeta=ABCMeta,
                     abstractmethod=abstractmethod, combinations=combinations, comb=comb,
                     _MetadataRequester=type('_MetadataRequester', (), {}),
                     metadata_routing=SimpleNamespace(UNUSED='unused'), indexable=indexable,
                     check_array=check_array, column_or_1d=lambda x: np.asarray(x).reshape(-1),
                     _num_samples=len, check_random_state=lambda s: s if isinstance(s, np.random.RandomState) else np.random.RandomState(s),
                     get_namespace=lambda *y, **kw: (np, False),
                     ceil=ceil, floor=floor, defaultdict=defaultdict, move_to=lambda *xs, **kw: np.asarray(xs[0]) if len(xs)==1 else tuple(np.asarray(x) for x in xs),
                     chain=chain, get_namespace_and_device=lambda x: (np, False, 'cpu'), _safe_indexing=lambda a, i: a[i])
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name in classes) or (isinstance(n, ast.FunctionDef) and n.name == '_validate_shuffle_split')]
    assert {n.name for n in nodes} == classes | {'_validate_shuffle_split'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    multiclass = ORACLE / 'sklearn/utils/multiclass.py'
    assert not subprocess.check_output(['git', '-C', str(ORACLE), 'diff', 'HEAD', '--', str(multiclass)], text=True)
    def finite(data, **kwargs):
        if not np.isfinite(data).all():
            raise ValueError('nonfinite target')
    target_ns = dict(np=np, Sequence=Sequence, issparse=issparse, warnings=warnings,
                     VisibleDeprecationWarning=np.exceptions.VisibleDeprecationWarning,
                     get_namespace=lambda *a, **kw: (np, False), check_array=lambda y, dtype=None, **kw: np.asarray(y, dtype=dtype),
                     cached_unique=lambda y, **kw: np.unique(y), _assert_all_finite=finite, _num_samples=len,
                     size=np.size, is_scalar_nan=lambda x: isinstance(x, numbers.Real) and np.isnan(x))
    kinds = [n for n in ast.parse(multiclass.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('_is_integral_float', 'is_multilabel', 'type_of_target')]
    exec(compile(ast.Module(body=kinds, type_ignores=[]), str(multiclass), 'exec'), target_ns)
    assert target_ns['type_of_target'].__code__.co_filename == str(multiclass)
    namespace['type_of_target'] = target_ns['type_of_target']
    extmath = ORACLE / 'sklearn/utils/extmath.py'
    mode = next(n for n in ast.parse(extmath.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_approximate_mode')
    exec(compile(ast.Module(body=[mode], type_ignores=[]), str(extmath), 'exec'), namespace)
    for function in ('check_cv', 'train_test_split'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        node.decorator_list = []  # Python parameter-validation decorator is binding scaffolding.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    for name in classes:
        for obj in namespace[name].__dict__.values():
            if hasattr(obj, '__code__'):
                assert obj.__code__.co_filename == str(source)

    cases = []
    def add(name, args, kwargs, n, groups=None, constructor=None, y=None):
        cv = namespace[name](*args, **kwargs)
        X = np.zeros((n, 1))
        folds = [(a.tolist(), b.tolist()) for a, b in cv.split(X, y=y, groups=groups)]
        assert len(folds) == cv.get_n_splits(X, y=y, groups=groups)
        cases.append((constructor or name + '(' + ', '.join(
            [str(x) for x in args] + [k + '=' + ('SplitSize(' + str(v) + ')' if k in ('test_size', 'train_size') and name in ('ShuffleSplit', 'GroupShuffleSplit', 'StratifiedShuffleSplit') else str(v)) for k, v in kwargs.items()]) + ')', n, groups, folds, y))

    for n in (2, 3, 8, 17):
        for k in (2, n):
            add('KFold', [k], {}, n)
            for seed in (0, 42):
                add('KFold', [k], dict(shuffle=True, random_state=seed), n)
        add('LeaveOneOut', [], {}, n)
        for p in (1, min(2, n-1)):
            add('LeavePOut', [p], {}, n)
    for n in (8, 13, 20):
        for gap in (0, 1):
            for maximum in (None, 3):
                kwargs = dict(gap=gap, test_size=2)
                if maximum is not None:
                    kwargs['max_train_size'] = maximum
                add('TimeSeriesSplit', [3], kwargs, n)
    for groups in ([2, -1, 2, 5, -1, 9], [0, 1, 2], [9, 9, -2, -2]):
        for name, args in [('LeaveOneGroupOut', []), ('LeavePGroupsOut', [1])]:
            add(name, args, {}, len(groups), np.array(groups))
    for labels in ([-1, 0, 1, 0, -2], [-1, -1], []):
        add('PredefinedSplit', [labels], {}, len(labels), constructor='PredefinedSplit(array(' + repr(labels) + '))')

    for n in (12, 19, 30):
        y = np.array(([8, -2, 5] * n)[:n])
        groups = np.array(([9, 2, -1, 4, 7, 9, 0] * n)[:n])
        for seed in (0, 42, 113):
            for shuffle in (False, True):
                options = dict(shuffle=shuffle)
                if shuffle: options['random_state'] = seed
                add('StratifiedKFold', [3], options, n, y=y)
                add('GroupKFold', [3], options, n, groups=groups)
                add('StratifiedGroupKFold', [3], options, n, groups=groups, y=y)
            add('RepeatedKFold', [], dict(n_splits=3, n_repeats=2, random_state=seed), n)
            add('RepeatedStratifiedKFold', [], dict(n_splits=3, n_repeats=2, random_state=seed), n, y=y)
            add('StratifiedShuffleSplit', [4], dict(random_state=seed, test_size=0.3), n, y=y)
            for name in ('ShuffleSplit', 'GroupShuffleSplit'):
                for sizes in ({}, dict(test_size=0.3), dict(train_size=2, test_size=2), dict(train_size=0.4)):
                    add(name, [3], dict(random_state=seed, **sizes), n, groups=groups if name=='GroupShuffleSplit' else None)

    fuzz = np.random.RandomState(123)
    for trial in range(40):
        n = int(fuzz.randint(12, 65))
        y = fuzz.randint(0, 3, n)
        groups = fuzz.randint(-3, 5, n)
        add('StratifiedGroupKFold', [3], dict(shuffle=True, random_state=trial), n, y=y, groups=groups)
        if min(np.bincount(y)) >= 2:
            add('StratifiedShuffleSplit', [2], dict(test_size=0.35, random_state=trial), n, y=y)
    for labels in ([7]*12, [9]*7 + [-2]*5 + [4]*3, [8]*10 + [-1]):
        add('StratifiedKFold', [3], dict(shuffle=True, random_state=3), len(labels), y=np.array(labels))

    add('StratifiedGroupKFold', [3], dict(shuffle=True, random_state=17), 12, y=np.array([7]*12), groups=np.arange(12)%4)
    for classifier in (False, True):
        labels = np.array([0]*5 + [2]*7)
        for shuffle in (False, True):
            kwargs = dict(classifier=classifier, shuffle=shuffle, random_state=42 if shuffle else None)
            cv = namespace['check_cv'](3, labels, **kwargs)
            expected = [(a.tolist(), b.tolist()) for a,b in cv.split(np.zeros((12,1)), labels)]
            constructor = 'check_cv(3, y=array(' + repr(labels.tolist()) + '), ' + ', '.join(k+'='+str(v) for k,v in kwargs.items()) + ')'
            cases.append((constructor, 12, None, expected, labels))
    for n in (12, 17):
        for shuffle in (False, True):
            for sizes in ({}, dict(test_size=0.3), dict(train_size=4, test_size=3), dict(train_size=0.6)):
                for stratify in (False, True) if shuffle else (False,):
                    labels = np.arange(n) % 2 if stratify else None
                    kwargs = dict(shuffle=shuffle, random_state=42, **sizes)
                    expected = namespace['train_test_split'](np.arange(n), stratify=labels, **kwargs)
                    args = ', '.join(k+'='+('SplitSize('+str(v)+')' if k in ('train_size','test_size') else str(v)) for k,v in kwargs.items())
                    if stratify: args += ', stratify=array('+repr(labels.tolist())+')'
                    constructor = 'train_test_split(array('+repr(list(range(n)))+'), '+args+')'
                    cases.append((constructor, n, None, [(expected[0].tolist(),expected[1].tolist())], None))

    def lit(values):
        """Native literal for a label/group array with the same dtype as the oracle input."""
        a = np.asarray(values)
        if a.dtype.kind == 'U':
            flat = '[' + ', '.join(json.dumps(str(v), ensure_ascii=False) for v in a.reshape(-1)) + ']'
            return f'strs({flat})' if a.ndim == 1 else f'SplitLabels(strs({flat}), {a.shape[0]}, {a.shape[1]})'
        kind = {'f': 'f64' if a.dtype == np.float64 else 'f32', 'i': 'i64'}[a.dtype.kind]
        flat = '[' + ', '.join(repr(float(v)) if a.dtype.kind == 'f' else str(int(v)) for v in a.reshape(-1)) + ']'
        shape = '' if a.ndim == 1 else f', {a.shape[0]}, {a.shape[1]}'
        call = f'{kind}({flat}{shape})'
        return f'SplitLabels({call})' if kind == 'i64' else call

    labeled = []
    def add_labeled(name, args, kwargs, n, y=None, groups=None, y_code=None, groups_code=None):
        cv = namespace[name](*args, **kwargs)
        X = np.zeros((n, 1))
        folds = [(a.tolist(), b.tolist()) for a, b in cv.split(X, y=y, groups=groups)]
        constructor = name + '(' + ', '.join([str(x) for x in args] + [k + '=' + ('SplitSize(' + str(v) + ')' if k in ('test_size', 'train_size') else str(v)) for k, v in kwargs.items()]) + ')'
        call_args = str(n)
        if y is not None: call_args += ', y=' + (y_code or lit(y))
        if groups is not None: call_args += ', groups=' + (groups_code or lit(groups))
        cases.append((constructor, n, None, folds, None, call_args))
        labeled.append(name)

    errors = []
    def add_error(name, args, kwargs, n, y=None, groups=None, fragment=''):
        cv = namespace[name](*args, **kwargs)
        try:
            list(cv.split(np.zeros((n, 1)), y=y, groups=groups))
        except ValueError as e:
            assert fragment in str(e), (fragment, str(e))
        else:
            raise AssertionError(f'reference accepted {name} {y!r}')
        call = str(n) + ('' if y is None else ', y=' + lit(y)) + ('' if groups is None else ', groups=' + lit(groups))
        errors.append((name + '(' + ', '.join([str(x) for x in args] + [k + '=' + str(v) for k, v in kwargs.items()]) + ')', call, fragment))

    words = np.array(['pear', 'apple', 'fig', 'apple', 'pear', 'fig', 'pear', 'kiwi', 'apple', 'pear', 'fig', 'kiwi'] * 2)
    numeric_words = np.array(['10', '9', '2', '10', '9', '2', '10', '2', '9', '10', '2', '9'] * 2)
    unicode = np.array(['é', 'e', 'Z', 'é', 'z', 'e', 'Z', 'é', 'z', 'e', 'Z', 'z'] * 2)
    floats = np.array([2.0, -1.0, 7.0, -1.0, 2.0, 7.0, 2.0, -1.0, 7.0, 2.0, -1.0, 7.0] * 2)
    fractional = np.array([0.5, -2.25, 0.5, 3.75, -2.25, 1e-3, 0.5, 3.75, 1e-3, -2.25, 0.5, 3.75] * 2)
    string_groups = np.array(['g9', 'g10', 'g2', 'g9', 'g7', 'g10', 'g2', 'g7', 'g9', 'g11', 'g2', 'g11'] * 2)
    for y in (words, numeric_words, unicode, floats, floats.astype(np.float32)):
        n = len(y)
        for shuffle in (False, True):
            options = dict(shuffle=shuffle)
            if shuffle: options['random_state'] = 7
            add_labeled('StratifiedKFold', [3], options, n, y=y)
            add_labeled('StratifiedGroupKFold', [3], options, n, y=y, groups=string_groups)
        add_labeled('RepeatedStratifiedKFold', [], dict(n_splits=3, n_repeats=2, random_state=5), n, y=y)
        add_labeled('StratifiedShuffleSplit', [3], dict(test_size=0.25, random_state=11), n, y=y)
    # StratifiedShuffleSplit accepts any finite labels, including fractional values.
    add_labeled('StratifiedShuffleSplit', [3], dict(test_size=0.25, random_state=3), len(fractional), y=fractional)
    for groups in (string_groups, fractional, fractional.astype(np.float32), numeric_words):
        n = len(groups)
        for shuffle in (False, True):
            options = dict(shuffle=shuffle)
            if shuffle: options['random_state'] = 13
            add_labeled('GroupKFold', [3], options, n, groups=groups)
        add_labeled('GroupShuffleSplit', [3], dict(test_size=0.3, random_state=17), n, groups=groups)
        add_labeled('LeaveOneGroupOut', [], {}, n, groups=groups)
        add_labeled('LeavePGroupsOut', [2], {}, n, groups=groups)
    # Mixed Int64 and non-Int64 labels/groups.
    int_y = np.array(([3, 1, 2] * 8)[:24])
    add_labeled('StratifiedGroupKFold', [3], dict(shuffle=True, random_state=1), 24, y=int_y, groups=string_groups, y_code='array(' + repr(int_y.tolist()) + ')')
    add_labeled('StratifiedGroupKFold', [3], {}, 24, y=words, groups=np.arange(24) % 6, groups_code='array(' + repr((np.arange(24) % 6).tolist()) + ')')
    # Rank-two StratifiedShuffleSplit labels join rows as strings and sort lexicographically.
    rows = np.array([[10, 0], [9, 1], [2, 0], [10, 0], [9, 1], [2, 0], [10, 0], [9, 1], [2, 0], [10, 0], [9, 1], [2, 0]])
    column = np.array([[10], [9], [2], [10], [9], [2], [10], [9], [2], [10], [9], [2]])
    for y in (rows, column, rows.astype(np.float64) / 4, np.array([['b', 'x'], ['a', 'y'], ['b', 'x'], ['a', 'y'], ['b', 'x'], ['a', 'y'], ['b', 'x'], ['a', 'y']])):
        add_labeled('StratifiedShuffleSplit', [4], dict(test_size=0.25, random_state=21), len(y), y=y)
    # Float64 row keys must match NumPy str() on varied magnitudes and signs.
    fuzz_keys = np.random.RandomState(99)
    for trial in range(12):
        base = fuzz_keys.choice([0.1, -0.0, 1e-5, 2.5e16, 123456.789, 1/3, 7.0, 1e300, -4.5e-7], size=(4, 2))
        y = np.repeat(base, 3, axis=0)
        add_labeled('StratifiedShuffleSplit', [2], dict(test_size=0.34, random_state=trial), len(y), y=y)
    # Reference validation errors.
    add_error('StratifiedKFold', [2], {}, 4, y=np.array([1.1, 1.2, 2.1, 2.2]), fragment='Supported target types')
    add_error('StratifiedGroupKFold', [2], {}, 4, y=np.array([0.5, 1.5, 0.5, 2.5]), groups=np.array(['a', 'b', 'c', 'd']), fragment='Supported target types')
    add_error('StratifiedKFold', [2], {}, 6, y=np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]), fragment='Supported target types')
    # check_cv dispatches on type_of_target, including continuous targets with classifier=True.
    for labels in (words, floats, fractional):
        for classifier in (False, True):
            cv = namespace['check_cv'](3, labels, classifier=classifier)
            expected = [(a.tolist(), b.tolist()) for a, b in cv.split(np.zeros((len(labels), 1)), labels)]
            cases.append((f'check_cv(3, {lit(labels)}, classifier={classifier})', len(labels), None, expected, None, f'{len(labels)}, y={lit(labels)}'))
    # train_test_split stratification with non-Int64 labels.
    for labels in (words, numeric_words, floats, fractional):
        n = len(labels)
        for sizes in ({}, dict(test_size=0.3), dict(train_size=12, test_size=6)):
            expected = namespace['train_test_split'](np.arange(n), stratify=labels, random_state=42, **sizes)
            args = ', '.join(k + '=SplitSize(' + str(v) + ')' for k, v in sizes.items())
            constructor = f'train_test_split(array({list(range(n))!r}), random_state=42, stratify={lit(labels)}' + (', ' + args if args else '') + ')'
            cases.append((constructor, n, None, [(expected[0].tolist(), expected[1].tolist())], None))

    code = '''from moxie.model_selection import check_cv, train_test_split, StratifiedGroupKFold, StratifiedShuffleSplit, StratifiedKFold, GroupKFold, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupShuffleSplit, SplitSize, KFold, TimeSeriesSplit, LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, PredefinedSplit, SplitLabels
from moxie.model_selection._indices import IndexArray, IndexSplit, SplitCursor
from moxie.model_selection._split import _SplitList
from numojo import NDArray, Shape, Item

def strs(values: List[String]) -> List[String]:
    return values.copy()

def numeric[dtype: DType](values: List[Float64], rows: Int, columns: Int) raises -> NDArray[dtype]:
    var result = NDArray[dtype](Shape(rows) if columns == 0 else Shape(rows, columns))
    for i in range(len(values)):
        if columns == 0: result[Item(i)] = values[i].cast[dtype]()
        else: result[Item(i // columns, i % columns)] = values[i].cast[dtype]()
    return result^

def f64(values: List[Float64]) raises -> NDArray[DType.float64]:
    return numeric[DType.float64](values, len(values), 0)

def f64(values: List[Float64], rows: Int, columns: Int) raises -> NDArray[DType.float64]:
    return numeric[DType.float64](values, rows, columns)

def f32(values: List[Float64]) raises -> NDArray[DType.float32]:
    return numeric[DType.float32](values, len(values), 0)

def i64(values: List[Int]) raises -> IndexArray:
    var result = IndexArray(Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^

def i64(values: List[Int], rows: Int, columns: Int) raises -> IndexArray:
    var result = IndexArray(Shape(rows, columns))
    for i in range(len(values)):
        result[Item(i // columns, i % columns)] = Int64(values[i])
    return result^

def array(values: List[Int]) raises -> IndexArray:
    var result = IndexArray(Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^

def emit[C: SplitCursor & Deinitable](var cursor: C, case_id: Int) raises:
    var fold = 0
    while True:
        var optional = cursor.next()
        if not optional:
            break
        var pair = optional.value().copy()
        for i in range(pair.train.size):
            print(case_id, fold, "train", Int(pair.train[Item(i)]))
        for i in range(pair.test.size):
            print(case_id, fold, "test", Int(pair.test[Item(i)]))
        print(case_id, fold, "end", 0)
        fold += 1

def main() raises:
'''
    for i, case in enumerate(cases):
        constructor, n, groups, _, y = case[:5]
        code += f'    var cv{i} = {constructor}\n'
        if len(case) == 6:
            code += f'    emit(cv{i}.split({case[5]}), {i})\n'
            continue
        if constructor.startswith('train_test_split('):
            code += f'    var pairs{i} = List[IndexSplit]()\n    pairs{i}.append(IndexSplit(cv{i}[0].copy(), cv{i}[1].copy()))\n    emit(_SplitList(pairs{i}^), {i})\n'
            continue
        group_arg = '' if groups is None else ', groups=array(' + repr(groups.tolist()) + ')'
        y_arg = '' if y is None else ', y=array(' + repr(y.tolist()) + ')'
        code += f'    emit(cv{i}.split({n}{y_arg}{group_arg}), {i})\n'
    for i, (constructor, call, fragment) in enumerate(errors):
        code += f'    try:\n        _ = {constructor}.split({call})\n        print("error", {i}, "accepted")\n    except e:\n        print("error", {i}, String(e).find({json.dumps(fragment)}) >= 0)\n'
    generated = ROOT / 'builds/parity_cases.mojo'
    generated.write_text(code)
    binary = ROOT / 'builds/parity_cases'
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', str(ROOT / 'builds/deps'), str(generated), '-o', str(binary)], check=True)
    actual = {}
    for line in subprocess.check_output([str(binary)], text=True).splitlines():
        if line.startswith('error '):
            _, index, verdict = line.split()
            assert verdict == 'True', ('native error mismatch', errors[int(index)], verdict)
            continue
        case, fold, part, value = line.split()
        pair = actual.setdefault((int(case), int(fold)), ([], []))
        if part != 'end':
            pair[part == 'test'].append(int(value))
    expected = {(i, j): pair for i, case in enumerate(cases) for j, pair in enumerate(case[3])}
    assert actual == expected, next(((key, actual.get(key), val) for key, val in expected.items() if actual.get(key) != val), 'Unexpected folds')
    exports = json.loads((ROOT / 'EXPORTS.json').read_text())['packages']['moxie.model_selection']
    missing = sorted(set(REQUIRED) - exports.keys())
    report = dict(reference=str(source), revision=REVISION,
                  sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cases=len(cases), folds=len(expected), tolerance=0, missing_exports=missing,
                  label_cases=len(labeled), error_cases=len(errors),
                  multiclass_sha256=hashlib.sha256(multiclass.read_bytes()).hexdigest())
    destination = ROOT / 'docs/private/splitters/parity.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS dense subset incl. String/Float labels: {len(cases)} cases ({len(labeled)} labeled splitter cases, {len(errors)} reference errors), {len(expected)} exact folds', flush=True)
    if missing:
        raise RuntimeError('Incomplete assigned public scope: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
