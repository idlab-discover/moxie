"""Compare native variance selection with the pinned sklearn dense fit body."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import warnings
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import feature_selection_checks as check

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def oracle():
    source = REFERENCE / 'sklearn/feature_selection/_variance_threshold.py'
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'status', '--porcelain', '--', source], text=True).strip()
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'fit', '_get_support_mask')]
    for method in methods:
        method.decorator_list = []
    node.bases = []
    node.body = methods
    def validate_data(estimator, X, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or 0 in X.shape or np.isinf(X).any():
            raise ValueError('invalid dense input')
        return X
    namespace = dict(np=np, validate_data=validate_data, check_is_fitted=lambda estimator: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['VarianceThreshold']


def literal(value):
    if np.isnan(value):
        return 'nan[DType.float64]()'
    if np.isposinf(value):
        return 'inf[DType.float64]()'
    if np.isneginf(value):
        return '-inf[DType.float64]()'
    return repr(float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    assembly_root = parser.parse_args().assembly_root.resolve()
    reference = oracle()
    rng = np.random.default_rng(417)
    fixtures = [
        np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3.]]),
        np.array([[0.], [1.]]),
        np.array([[0., 1], [20., 1]]),
        np.full((10, 2), -0.13725701),
        np.array([[np.nan, 0., np.nan], [np.nan, 1., 3], [np.nan, 2., 3]]),
        np.full((3, 2), np.nan),
        np.array([[1., 2.]]),
        np.array([[np.inf, 0.], [0., 1.]]),
        rng.normal(size=(17, 5)),
        1e10 + rng.integers(-10, 10, size=(12, 3)),
    ]
    fixtures += [
        # Exact binary variance and its adjacent threshold values.
        np.array([[0., 0., 7.], [1., 2., 7.]]),
        # Squaring overflows; the zero-threshold range cap remains finite.
        np.array([[-1e200, 0.], [1e200, 2.]]),
        np.array([[1e308, 0.], [1e308, 2.]]),
        # Squared differences underflow in Float64.
        np.array([[0., 0.], [1e-200, 2.]]),
        # NaNs disable the range cap for that column, even with large spread.
        np.array([[0., 0., np.nan], [20., 2., 5.], [np.nan, 4., np.nan]]),
    ]
    lines = ['import numojo as nm', 'from std.math import nan, inf',
             'from moxie.feature_selection import VarianceThreshold']
    expected, exact, calls, bodies, metadata = [], [], [], [], []
    header = lines.copy()
    def emit(expression, value, is_exact=False):
        index = len(expected)
        expected.append(float(value))
        if is_exact:
            exact.append(index)
        lines.append(f'    print({index}, {expression})')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dtype in (np.float32, np.float64):
            dt = np.dtype(dtype).name
            for fixture_index, fixture in enumerate(fixtures):
                thresholds = (0., .25, 1.)
                if fixture_index == 10:
                    thresholds += (np.nextafter(.25, 0.), np.nextafter(.25, 1.))
                for threshold in thresholds:
                    for layout in ("contiguous", "reverse", "padded"):
                        lines = []
                        name = f'case{len(calls)}'
                        calls.append(name)
                        bodies.append(lines)
                        metadata.append(dict(name=name, fixture=fixture_index, dtype=dt,
                                             threshold=float(threshold), layout=layout,
                                             first_comparison=len(expected)))
                        lines += ['@no_inline', f'def {name}() raises:']
                        raw = fixture.astype(dtype)
                        X = raw[::-1, ::-1] if layout == "reverse" else raw
                        rows, cols = raw.shape
                        row_stride = 2 * cols + 3 if layout == "padded" else cols
                        col_stride = 2 if layout == "padded" else 1
                        offset = 3 if layout == "padded" else 0
                        size = offset + rows * row_stride
                        # NaN padding detects accidental inclusion in reductions.
                        lines.append(f'    var backing = nm.full[DType.{dt}](nm.Shape({size}), nan[DType.{dt}]())')
                        for i in range(rows):
                            for j in range(cols):
                                position = offset + i * row_stride + j * col_stride
                                lines.append(f'    backing.unsafe_set({position}, SIMD[DType.{dt}, 1]({literal(raw[i, j])}))')
                        if layout == "reverse":
                            offset = raw.size - 1
                            row_stride, col_stride = -cols, -1
                        lines.append(f'    var X = backing.view_with_layout(nm.Shape({rows}, {cols}), nm.NDArrayStrides({row_stride}, {col_stride}), {offset})')
                        lines.append(f'    var selector = VarianceThreshold[DType.{dt}]({threshold})')
                        estimator = reference(threshold)
                        try:
                            estimator.fit(X)
                        except ValueError:
                            lines += ['    var failed = False', '    try:', '        _ = selector.fit(X)', '    except err:', '        failed = String(err).startswith("ValueError:")']
                            emit('Int(failed)', 1, True)
                            continue
                        lines += ['    _ = selector.fit(X)', '    var variances: nm.NDArray[DType.float64] = selector.variances_()',
                                  '    var mask = selector.get_support()[List[Bool]].copy()',
                                  f'    var output: nm.NDArray[DType.{dt}] = selector.transform(X)']
                        mask = estimator._get_support_mask()
                        for j, value in enumerate(estimator.variances_):
                            emit(f'variances.unsafe_get({j})', value)
                            emit(f'Int(mask[{j}])', mask[j], True)
                        emit('output.shape[1]', mask.sum(), True)
                        for j, value in enumerate(X[:, mask].ravel()):
                            emit(f'Float64(output.unsafe_get({j}))', value, True)
    folder = ROOT / 'builds/variance-parity-source'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'expected.json').write_text(json.dumps(dict(
        reference=REVISION, fixtures=metadata, values=expected), indent=2) + '\n')
    outputs = []
    # Bound optimizer work as coverage grows; every batch retains -O3 and
    # the shared compiler lock, memory ceiling, and timeout.
    for start in range(0, len(calls), 24):
        batch = calls[start:start + 24]
        lines = header + [line for body in bodies[start:start + 24] for line in body]
        lines += ['def main() raises:'] + [f'    {name}()' for name in batch]
        source = folder / f'batch{start // 24}.mojo'
        source.write_text('\n'.join(lines) + '\n')
        check.compile_file(assembly_root, source, 'variance-parity')
        result = check.run(assembly_root / 'builds/variance-parity', capture=True)
        outputs.extend(line.split() for line in result.stdout.splitlines())
    (folder / 'native.txt').write_text('\n'.join(' '.join(row) for row in outputs) + '\n')
    # Verify output identity as well as count so omitted/reordered rows cannot pass.
    np.testing.assert_array_equal([int(row[0]) for row in outputs], np.arange(len(expected)))
    actual = np.array([float(row[1]) for row in outputs])
    expected = np.array(expected)
    np.testing.assert_array_equal(actual[exact], expected[exact])
    np.testing.assert_array_equal(actual[expected == 0], expected[expected == 0])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    (ROOT / 'builds/variance-parity.json').write_text(json.dumps(dict(
        reference=REVISION, cases=len(calls), comparisons=len(expected),
        rtol=1e-10, atol=1e-12, fixtures=metadata), indent=2) + '\n')
    print('PASS', len(calls), 'variance cases;', len(expected), 'comparisons')


if __name__ == '__main__':
    main()
