"""Integer specialization compared with the pinned local sklearn estimator."""
from pathlib import Path
import argparse
import json
import sys
import warnings
import numpy as np
from oracle import load
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, default=ROOT / 'builds')
builds = parser.parse_args().build_dir.resolve()
check.BUILDS = builds
oracle = load()
lines = ['''import numojo as nm
from moxie.impute import IntegerSimpleImputer
from moxie.impute._plumbing import MissingValue
from moxie.impute._simple import ColumnStrategy

struct Minimum(ColumnStrategy):
    def __init__(out self):
        pass
    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        if values.size == 0:
            return 17
        return nm.min(values)

@no_inline
def emit[dtype: DType](name: String, X: nm.NDArray[dtype]) raises:
    print("R", name, X.shape[0] if X.ndim == 2 else 1, X.shape[1] if X.ndim == 2 else X.size)
    for i in range(X.size):
        print(X.unsafe_get(i))

@no_inline
def exercise[dtype: DType](name: String, X: nm.NDArray[dtype], T: nm.NDArray[dtype], strategy: String, marker: MissingValue[dtype], fill: SIMD[dtype, 1], keep: Bool, indicator: Bool) raises:
    var model = IntegerSimpleImputer[dtype, Minimum](strategy=strategy, callback=Minimum(), missing_values=marker, fill_value=fill, keep_empty_features=keep, add_indicator=indicator)
    model.fit(X)
    emit(name + "s", model.statistics_)
    var output = model.transform(T)
    if strategy == "mean" or strategy == "median" or strategy == "callable":
        emit(name + "o", output[nm.NDArray[DType.float64]])
    else:
        emit(name + "o", output[nm.NDArray[dtype]])
    if indicator:
        try:
            emit(name + "i", model.inverse_transform(output))
        except:
            print("E", name + "i")
''']
expected = {}
cases = 0
rng = np.random.default_rng(841)


def array(name, data, dtype):
    lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({data.shape[0]}, {data.shape[1]}))')
    for i, value in enumerate(data.flat):
        lines.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({int(value)}))')


def record(name, value):
    expected[name] = None if value is None else np.asarray(value).reshape(1, -1) if np.asarray(value).ndim == 1 else np.asarray(value)


def case(X, T, strategy, marker, fill, keep, indicator):
    global cases
    dtype = X.dtype.name
    callback = lambda values: np.min(values) if len(values) else 17
    model = oracle.SimpleImputer(strategy=callback if strategy == 'callable' else strategy, missing_values=marker, fill_value=X.dtype.type(fill), keep_empty_features=keep, add_indicator=indicator).fit(X)
    out = model.transform(T)
    assert out.dtype == (np.dtype('float64') if strategy in ('mean', 'median', 'callable') else X.dtype)
    name = f'c{cases}'
    record(name + 's', np.asarray(model.statistics_, dtype=np.float64))
    record(name + 'o', out)
    if indicator:
        try:
            inverse = model.inverse_transform(out)
        except IndexError:
            inverse = None
        record(name + 'i', inverse)
    lines.extend(['@no_inline', f'def case{cases}() raises:'])
    array('X', X, dtype)
    array('T', T, dtype)
    marker_arg = f'MissingValue[DType.{dtype}]()' if np.isnan(marker) else f'MissingValue[DType.{dtype}](SIMD[DType.{dtype}, 1]({int(marker)}))'
    lines.append(f'    exercise("{name}", X, T, "{strategy}", {marker_arg}, SIMD[DType.{dtype}, 1]({int(fill)}), {keep}, {indicator})')
    cases += 1


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for dtype in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
        for strategy in ('mean', 'median', 'most_frequent', 'constant', 'callable'):
            for keep in (False, True):
                for indicator in (False, True):
                    for marker in (0, np.nan):
                        X = rng.integers(1, 9, size=(7, 4)).astype(dtype)
                        if not np.isnan(marker):
                            X[rng.random(X.shape) < .35] = marker
                            X[:, 2] = marker
                        X[:, 3] = 2
                        T = X[:3].copy()
                        T[0, 3] = 0
                        case(X, T, strategy, marker, 7, keep, indicator)
        # Exact sentinel equality, ordering and constant fills above 2**53.
        info = np.iinfo(dtype)
        for fill in (info.min, info.max):
            X = np.array([[fill, 0], [0, 0]], dtype=dtype)
            case(X, X, 'constant', 0, fill, True, False)
        for strategy in ('mean', 'median', 'callable'):
            X = np.array([[info.max], [info.max - 1], [0]], dtype=dtype)
            case(X, X, strategy, 0, 7, False, True)
        if dtype.startswith('int'):
            X = np.array([[-9, -1], [-5, -1], [-9, 2], [-1, 4]], dtype=dtype)
            for strategy in ('mean', 'median', 'most_frequent', 'constant', 'callable'):
                case(X, X, strategy, -1, -7, False, True)
        X = np.zeros((2, 3), dtype=dtype)
        for strategy in ('mean', 'median', 'most_frequent', 'constant', 'callable'):
            case(X, X, strategy, 0, 7, False, True)
    for dtype in ('int64', 'uint64'):
        for strategy in ('mean', 'median', 'most_frequent', 'constant', 'callable'):
            # Promotion before masking merges adjacent large integer values.
            X = np.array([[2**53, 4], [2**53 + 1, 6], [2**53 + 3, 0]], dtype=dtype)
            case(X, X, strategy, 2**53, 2**53 + 1, False, True)
            # Mode stores Float64 statistics even though output remains integer.
            X = np.array([[2**53 + 1], [2**53 + 1], [0]], dtype=dtype)
            case(X, X, strategy, 0, 2**53 + 1, False, False)
lines.append('def main() raises:')
lines.extend(f'    case{i}()' for i in range(cases))
source = builds / 'integer_parity.mojo'
source.write_text('\n'.join(lines) + '\n')
check.compile(source, builds / 'integer_parity')
result = check.run(builds / 'integer_parity', capture_output=True, text=True)
(builds / 'integer-parity-output.txt').write_text(result.stdout)
output = iter(result.stdout.splitlines())
seen = set()
for line in output:
    header = line.split()
    assert header[0] in ('R', 'E'), line
    name = header[1]
    assert name in expected and name not in seen, name
    seen.add(name)
    wanted = expected[name]
    if header[0] == 'E':
        assert wanted is None, name
        continue
    assert wanted is not None, name
    rows, cols = map(int, header[2:])
    integer = wanted.dtype.kind in 'iu'
    values = np.array([(int if integer else float)(next(output)) for _ in range(rows * cols)], dtype=wanted.dtype).reshape(rows, cols)
    assert values.shape == wanted.shape, name
    if integer:
        np.testing.assert_array_equal(values, wanted, err_msg=name)
    else:
        np.testing.assert_allclose(values, wanted, rtol=3e-13, atol=3e-13, equal_nan=True, err_msg=name)
assert seen == set(expected)
report = dict(cases=cases, records=len(seen), seed=841, integer_comparison='exact', float64_rtol=3e-13, scope='same-dtype integer inputs; mode statistics rounding outside 64-bit range raises explicitly')
(ROOT / 'docs/private/impute.simple/integer-parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS: {cases} integer sklearn differential cases, {len(seen)} records')
