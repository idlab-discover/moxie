"""Exact categorical identity comparison with pinned local sklearn source.

Only NumPy array-namespace plumbing is supplied by this harness. Missing
categories are mapped to Moxie's explicit -1 storage tag after reference encoding.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/utils/_encode.py', 'sklearn/utils/_missing.py',
         'sklearn/utils/tests/test_encode.py', 'sklearn/preprocessing/_encoders.py']


def oracle():
    assert subprocess.check_output(['git', '-C', SOURCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', SOURCE, 'status', '--porcelain', '--', *FILES], text=True)
    namespace = {}
    path = SOURCE / FILES[1]
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    path = SOURCE / FILES[0]
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in
        ('sklearn.utils._array_api', 'sklearn.utils._missing'))]
    namespace.update(get_namespace=lambda *args: (np, False),
                     array_device=lambda value: 'cpu', size=np.size)
    exec(compile(tree, str(path), 'exec'), namespace)
    assert namespace['_unique'].__code__.co_filename == str(path)
    return namespace


def literal(value, dtype):
    if dtype == 'String':
        return json.dumps(value, ensure_ascii=False)
    if dtype == 'bool':
        return 'True' if value else 'False'
    if dtype == 'int64':
        return str(int(value))
    if np.isnan(value):
        return f'nan[DType.{dtype}]()'
    return repr(float(value))


def main():
    reference = oracle()
    cases = [
        ('String', ['z', None, '', 'é', 'z', 'e\u0301', '😀', 'None']),
        ('String', []), ('String', [None, None]),
        ('String', ['b', 'a', 'b']),
        ('int64', [2**53 + 1, 2**53, -(2**63), 2**63 - 1]),
        ('bool', [True, False, True]),
    ]
    rng = np.random.default_rng(42)
    for dtype in ['int64', 'bool', 'float32', 'float64']:
        for length in [0, 1, 7, 64]:
            for _ in range(3):
                values = rng.integers(-4, 5, size=length).astype(dtype)
                if dtype.startswith('float'):
                    values /= 4
                    values[::3] = np.nan
                cases.append((dtype, values.tolist()))
    lines = [
        'from moxie._categorical import categorical_column',
        'from std.testing import assert_equal, assert_false',
        'from std.math import nan',
        'from numojo import Shape, Item', 'import numojo as nm', '',
    ]
    for number, (dtype, values) in enumerate(cases):
        array = np.asarray(values, dtype=object if dtype == 'String' else dtype)
        if len(array):
            classes, codes = reference['_unique'](array, return_inverse=True)
        else:
            # Pinned object helper cannot request inverse codes for empty input.
            classes = reference['_unique'](array)
            codes = np.empty(0, dtype=np.int64)
        retained = [i for i, value in enumerate(classes)
                    if value is not None and not reference['is_scalar_nan'](value)]
        remap = {old: new for new, old in enumerate(retained)}
        expected_codes = [remap.get(int(code), -1) for code in codes]
        expected_classes = classes[retained]
        lines.append(f'def case_{number}() raises:')
        if dtype == 'String':
            cells = ['None' if value is None else f'Optional[String]({literal(value, dtype)})' for value in values]
            lines.append('    var input = List[Optional[String]]()')
            lines.extend(f'    input.append({cell})' for cell in cells)
        else:
            lines.append(f'    var input = nm.zeros[DType.{dtype}](Shape({len(values)}))')
            lines.extend(f'    input[Item({i})] = {literal(value, dtype)}' for i, value in enumerate(values))
        lines += ['    var column = categorical_column(input)',
                  f'    assert_equal(column.rows(), {len(values)})',
                  f'    assert_equal(column.domain().size(), {len(expected_classes)})']
        for i, value in enumerate(expected_classes):
            expected = literal(value, dtype)
            if dtype != 'String':
                expected = f'Scalar[DType.{dtype}]({expected})'
            lines.append(f'    assert_equal(column.domain().get({i}), {expected})')
        for i, code in enumerate(expected_codes):
            lines.append(f'    assert_equal(column.codes()[Item({i})], Int64({code}))')
            if code == -1:
                lines.append(f'    assert_false(column.get({i}))')
            else:
                expected = literal(values[i], dtype)
                if dtype != 'String':
                    expected = f'Scalar[DType.{dtype}]({expected})'
                lines.append(f'    assert_equal(column.get({i}).value(), {expected})')
        lines.append('')
    lines += ['def main() raises:'] + [f'    case_{i}()' for i in range(len(cases))]
    directory = ROOT / 'builds/categorical'
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'parity.mojo'
    source.write_text('\n'.join(lines) + '\n')
    binary = directory / 'parity'
    subprocess.run(['mojo', 'build', '-I', str(ROOT / 'builds/deps'), str(source), '-o', str(binary)], check=True)
    subprocess.run([binary], check=True)
    (directory / 'reference.json').write_text(json.dumps({
        'revision': REVISION, 'numpy': np.__version__, 'cases': len(cases),
        'files': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in FILES},
    }, indent=2) + '\n')
    print(f'PASS {len(cases)} categorical pinned-source comparisons (exact)')


if __name__ == '__main__':
    main()
