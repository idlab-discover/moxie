"""Exact dense helper parity against original pinned sklearn function bodies.

The harness supplies NumPy array namespace and validation plumbing only. No
classification or encoding algorithm is replaced by a test double.
"""
from collections.abc import Sequence
import ast
import hashlib
import json
import numbers
from pathlib import Path
import subprocess
import warnings

import numpy as np
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parents[1]
SKLEARN = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/utils/multiclass.py', 'sklearn/utils/_encode.py',
         'sklearn/utils/tests/test_multiclass.py', 'sklearn/utils/tests/test_encode.py',
         'sklearn/model_selection/_split.py', 'sklearn/neighbors/_base.py',
         'sklearn/preprocessing/_label.py', 'doc/modules/preprocessing_targets.rst']


def oracle():
    assert subprocess.check_output(['git', '-C', SKLEARN, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', SKLEARN, 'diff', 'HEAD', '--', *FILES], text=True)

    def check_array(y, dtype=None, **kwargs):
        return np.asarray(y, dtype=dtype)

    def finite(data, **kwargs):
        if not np.isfinite(data).all():
            raise ValueError('nonfinite target')

    ns = dict(np=np, Sequence=Sequence, issparse=issparse, warnings=warnings,
              VisibleDeprecationWarning=np.exceptions.VisibleDeprecationWarning,
              get_namespace=lambda *a: (np, False), check_array=check_array,
              cached_unique=lambda y, **kw: np.unique(y), _assert_all_finite=finite,
              _num_samples=len, size=np.size,
              is_scalar_nan=lambda x: isinstance(x, numbers.Real) and np.isnan(x))
    for filename, names in [
        ('sklearn/utils/multiclass.py', {'_is_integral_float', 'is_multilabel', 'type_of_target', 'check_classification_targets'}),
        ('sklearn/utils/_encode.py', {'_unique', '_unique_np'}),
    ]:
        path = SKLEARN / filename
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        for name in names:
            assert ns[name].__code__.co_filename == str(path)
    return ns


def literal(x, dtype):
    if dtype == 'String':
        return json.dumps(str(x), ensure_ascii=False)
    if dtype == 'Int64':
        return f'Int64({int(x)})'
    return f'{dtype}({float(x)!r})'


def values(xs, dtype):
    return 'List[' + dtype + ']()' if not len(xs) else '[' + ', '.join(literal(x, dtype) for x in xs) + ']'


HEADER = '''from moxie._labels import StringTarget, encode_column, type_of_target
from numojo import Shape, Item
import numojo as nm
from std.testing import assert_equal, assert_raises


def check_index(actual: nm.NDArray[DType.int64], expected: List[Int64]) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.size, len(expected))
    for i in range(len(expected)):
        assert_equal(actual[Item(i)], expected[i])


def numeric[dtype: DType](values: List[Scalar[dtype]], rows: Int, columns: Int) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](Shape(rows)) if columns == -1 else nm.zeros[dtype](Shape(rows, columns))
    for i in range(len(values)):
        if columns == -1:
            result[Item(i)] = values[i]
        else:
            result[Item(i // columns, i % columns)] = values[i]
    return result^

'''


def main():
    ns = oracle()
    rng = np.random.RandomState(849)
    cases = []
    for dtype in (np.int64, np.float32, np.float64):
        arrays = [[], [9], [9, 2, 9, -1], [2, 1], [0, 0, 0], [[1], [2]],
                  [[0, 1], [1, 0]], [[-3, 3]], [[1, 2], [3, 1]],
                  [[7, 7], [7, 7]], [[1, 0, 2]], [0, -0.0]]
        for a in arrays:
            cases.append((np.array(a, dtype=dtype), True))
        for shape in ((0,), (0, 1), (0, 2), (1, 0), (0, 0)):
            cases.append((np.empty(shape, dtype=dtype), True))
        for shape in ((17,), (23, 1), (23, 3), (7, 4)):
            for _ in range(3):
                cases.append((rng.randint(-5, 9, size=shape).astype(dtype), True))
        if dtype != np.int64:
            cases.extend([(np.array(a, dtype=dtype), False) for a in
                          [[1.25, 1.5, 1.25], [1.0, 1.0001, 2.0], [[1.25, 2], [3, 4.5]],
                           [2.0**63], [-2.0**63], [2.0**64]]])
        else:
            cases.append((np.array([-(2**63), 2**63-1, 2**53+1, 2**53, -(2**63)], dtype=dtype), True))
    for a in [[], ['z', 'a', 'z'], ['', ' ', 'a'], ['é', 'é', '😀', 'Ω', 'a', 'é'],
              [['z'], ['a']], [['z', 'one'], ['a', 'two']], [['x', 'x']],
              ['10', '2', '1'], ['a b', 'a', 'b']]:
        cases.append((np.array(a, dtype=str), True))
    for shape in ((0,), (0, 1), (0, 2), (1, 0)):
        cases.append((np.empty(shape, dtype=str), True))

    parts = [HEADER]
    count = 0
    for case, (y, classification) in enumerate(cases):
        is_string = y.dtype.kind == 'U'
        dtype = 'String' if is_string else {'i': 'Int64', 'f': 'Float32' if y.dtype.itemsize == 4 else 'Float64'}[y.dtype.kind]
        mdtype = {'Int64': 'DType.int64', 'Float32': 'DType.float32', 'Float64': 'DType.float64'}.get(dtype)
        rows, columns = y.shape[0], (-1 if y.ndim == 1 else y.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kind = ns['type_of_target'](y)
            try:
                ns['check_classification_targets'](y)
                valid = True
            except ValueError:
                valid = False
        body = [f'def case_{case}() raises:']
        if is_string:
            arguments = values(y.ravel(), dtype) + (f', {rows}, {columns}' if y.ndim == 2 else '')
            body.append(f'    var y = StringTarget({arguments})')
        else:
            body.append(f'    var y = numeric[{mdtype}]({values(y.ravel(), dtype)}, {rows}, {columns})')
        body.append(f'    assert_equal(type_of_target(y), {json.dumps(kind)})')
        for col in range(1 if columns == -1 else columns):
            if classification and not valid:
                body.extend(['    with assert_raises():', f'        _ = encode_column(y, {col})'])
                continue
            column = y if columns == -1 else y[:, col]
            classes, inverse, counts = ns['_unique'](column, return_inverse=True, return_counts=True)
            np_classes, first = np.unique(column, return_index=True)
            np.testing.assert_array_equal(classes, np_classes)
            appearance = np.argsort(first)
            ranks = np.empty(len(first), dtype=np.int64)
            ranks[appearance] = np.arange(len(first))
            body.append(f'    var e{col} = encode_column(y, {col}, classification={classification})')
            body.append(f'    var expected{col}: List[{dtype}] = {values(classes, dtype)}')
            body.append(f'    assert_equal({"len(e"+str(col)+".classes)" if is_string else "e"+str(col)+".classes.size"}, len(expected{col}))')
            body.append(f'    for i in range(len(expected{col})):')
            access = '[i]' if is_string else '[Item(i)]'
            body.append(f'        assert_equal(e{col}.classes{access}, expected{col}[i])')
            for name, expected in [('inverse', inverse), ('counts', counts), ('first_indices', first), ('appearance_order', appearance)]:
                body.append(f'    check_index(e{col}.index.{name}, {values(expected, "Int64")})')
            body.append(f'    check_index(e{col}.index.first_seen_codes(), {values(ranks[inverse], "Int64")})')
            body.append(f'    check_index(e{col}.encode(y, {col}), {values(inverse, "Int64")})')
            body.append(f'    var decoded{col} = e{col}.decode(e{col}.index.inverse)')
            body.append(f'    var original{col}: List[{dtype}] = {values(column, dtype)}')
            body.append(f'    for i in range(len(original{col})):')
            decoded = f'decoded{col}.get(i)' if is_string else f'decoded{col}[Item(i)]'
            body.append(f'        assert_equal({decoded}, original{col}[i])')
            count += 1
        parts.append('\n'.join(body) + '\n\n')
    parts.append('def main() raises:\n' + ''.join(f'    case_{i}()\n' for i in range(len(cases))) + f'    print("PASS {len(cases)} target-kind cases and {count} exact column-domain comparisons")\n')
    source = ROOT / 'builds/parity.mojo'
    source.write_text(''.join(parts))
    subprocess.run(['mojo', 'build', '-O3', '-I', str(ROOT/'builds/deps'), str(source), '-o', str(ROOT/'builds/parity')], check=True)
    subprocess.run([ROOT/'builds/parity'], check=True)
    (ROOT/'builds/parity-report.json').write_text(json.dumps(dict(reference=REVISION,
        target_cases=len(cases), column_cases=count, numpy=np.__version__,
        source_hashes={f: hashlib.sha256((SKLEARN/f).read_bytes()).hexdigest() for f in FILES}), indent=2)+'\n')


if __name__ == '__main__':
    main()
