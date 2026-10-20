"""Exact text, missingness and Boolean parity against pinned sklearn bodies."""
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
from moxie.impute import StringMatrix, StringSimpleImputer, StringMissingIndicator

@no_inline
def emit(name: String, cells: List[Optional[String]], rows: Int, columns: Int) raises:
    print("S", name, rows, columns)
    for cell in cells:
        if cell:
            print("V", cell.value().count_codepoints())
            for cp in cell.value().codepoints():
                print(Int(cp))
        else:
            print("N")

@no_inline
def exercise(name: String, X: StringMatrix, T: StringMatrix, strategy: String, marker: Optional[String], fill: Optional[String], keep: Bool, indicator: Bool) raises:
    var model = StringSimpleImputer(strategy=strategy, missing_values=marker, fill_value=fill, keep_empty_features=keep, add_indicator=indicator)
    var names = List[String]()
    for j in range(X.columns()):
        names.append("field" + String(j))
    try:
        model.fit_named(X, names)
        var result = model.transform(T)
        emit(name + "s", model.statistics_, 1, model.n_features_in_)
        emit(name + "v", result.values.values(), result.values.rows(), result.values.columns())
        print("B", name + "b", result.indicators.shape[0], result.indicators.shape[1])
        for i in range(result.indicators.size):
            print(Int(result.indicators.unsafe_get(i)))
        var output_names = model.get_feature_names_out()
        var cells = List[Optional[String]]()
        for value in output_names:
            cells.append(Optional[String](value.copy()))
        emit(name + "n", cells, 1, len(cells))
    except:
        print("E", name)
''']
lines.append('''
@no_inline
def exercise_indicator(name: String, X: StringMatrix, T: StringMatrix, marker: Optional[String], features: String, error_on_new: Bool) raises:
    var model = StringMissingIndicator(missing_values=marker, features=features, error_on_new=error_on_new)
    try:
        _ = model.fit(X)
        var result = model.transform(T)
        print("B", name, result.shape[0], result.shape[1])
        for i in range(result.size):
            print(Int(result.unsafe_get(i)))
        var names = model.get_feature_names_out()
        var cells = List[Optional[String]]()
        for value in names:
            cells.append(Optional[String](value.copy()))
        emit(name + "n", cells, 1, len(cells))
    except:
        print("E", name)
''')
expected = {}
cases = 0
rng = np.random.default_rng(317)


def string(value):
    # Mojo string escape syntax follows JSON for these explicit controls.
    if value is None:
        return 'None'
    return 'String(' + json.dumps(str(value), ensure_ascii=False) + ')'


def matrix(name, value):
    vals = ', '.join(string(v) for v in value.flat)
    lines.append(f'    var {name}_cells: List[Optional[String]] = [{vals}]')
    lines.append(f'    var {name} = StringMatrix({name}_cells, {value.shape[0]}, {value.shape[1]})')


def oracle_input(data):
    return np.array([[np.nan if v is None else v for v in row] for row in data], dtype=object)


def normalized(data):
    return np.array([None if not isinstance(v, str) else v for v in np.asarray(data).flat], dtype=object).reshape(np.asarray(data).shape)


def case(X, T, strategy, marker, fill, keep, indicator):
    global cases
    name = f'c{cases}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = oracle.SimpleImputer(strategy=strategy, missing_values=np.nan if marker is None else marker, fill_value=fill, keep_empty_features=keep, add_indicator=indicator)
        try:
            model.fit(oracle_input(X))
            result = model.transform(oracle_input(T))
        except ValueError:
            expected[name] = None
        else:
            if strategy == 'most_frequent' and any(not isinstance(v, str) for v in model.statistics_ if not (isinstance(v, float) and np.isnan(v))):
                # The reference fills retained empty object columns with numeric 0.
                assert keep and any(v == 0 for v in model.statistics_)
                expected[name] = None
            else:
                expected[name + 's'] = normalized(model.statistics_).reshape(1, -1)
                k = len(model.indicator_.features_) if indicator else 0
                width = result.shape[1] - k
                expected[name + 'v'] = normalized(result[:, :width])
                expected[name + 'b'] = np.asarray(result[:, width:], dtype=bool)
                expected[name + 'n'] = model.get_feature_names_out([f'field{j}' for j in range(X.shape[1])]).reshape(1, -1)
    lines.extend(['@no_inline', f'def case{cases}() raises:'])
    matrix('X', X)
    matrix('T', T)
    lines.append(f'    exercise("{name}", X, T, {string(strategy)}, {string(marker)}, {string(fill)}, {keep}, {indicator})')
    cases += 1


alphabet = np.array(['red', 'blue', '', 'é', 'é', '🐈', '中', 'Z', 'line\nbreak', 'a\tb'], dtype=object)
for strategy in ('most_frequent', 'constant'):
    for marker in (None, '', '<MISSING>'):
        for keep in (False, True):
            for indicator in (False, True):
                for empty_column in (False, True):
                    for fill in (None, '', 'replacement🐈'):
                        X = rng.choice(alphabet, size=(7, 4)).copy()
                        X[rng.random(X.shape) < .4] = marker
                        X[0, :] = 'observed'
                        if empty_column:
                            X[:, 2] = marker
                        X[:, 3] = 'complete'
                        T = X[:3].copy()
                        T[0, 3] = marker
                        case(X, T, strategy, marker, fill, keep, indicator)
# Deliberate lexicographic ties, repeated modes, empty text and numeric-looking text.
for values in (['z', 'a', None], ['é', 'é', None], ['🐈', '中', None], ['', 'a', None], ['10', '2', None], ['a\x00b', 'a', None]):
    X = np.array(values, dtype=object).reshape(-1, 1)
    case(X, X, 'most_frequent', None, None, False, True)
for strategy in ('most_frequent', 'constant'):
    X = np.full((2, 3), None, dtype=object)
    for keep in (False, True):
        case(X, X, strategy, None, None, keep, True)
for strategy in ('mean', 'median', 'invalid'):
    X = np.array([['red'], [None]], dtype=object)
    case(X, X, strategy, None, None, False, False)
X = np.array([['red'], [None]], dtype=object)
case(X, X, 'constant', '<MISSING>', 'fill', False, False)
# Confirm why string inverse is explicitly unavailable in the native API.
inverse_model = oracle.SimpleImputer(strategy='most_frequent', add_indicator=True)
inverse_output = inverse_model.fit_transform(np.array([['red'], [np.nan]], dtype=object))
try:
    inverse_model.inverse_transform(inverse_output)
except ValueError:
    pass
else:
    raise AssertionError('Pinned reference unexpectedly supports arbitrary string inverse')
# Standalone categorical indicators, including new missingness and empty outputs.
indicator_cases = 0
for marker in (None, '', '<MISSING>', '🐈'):
    for features in ('missing-only', 'all'):
        for error_on_new in (False, True):
            for kind in ('mixed', 'complete', 'empty', 'invalid_marker'):
                X = np.array([['a', marker, 'é'], ['b', 'z', '中']], dtype=object)
                if kind == 'complete':
                    X[:] = 'observed'
                elif kind == 'empty':
                    X[:] = marker
                elif kind == 'invalid_marker':
                    X[0, 0] = None
                T = X.copy()
                T[1, 0] = marker
                name = f'c{cases}'
                model = oracle.MissingIndicator(missing_values=np.nan if marker is None else marker,
                                                features=features, error_on_new=error_on_new)
                try:
                    model.fit(oracle_input(X))
                    result = model.transform(oracle_input(T))
                except ValueError:
                    expected[name] = None
                else:
                    expected[name] = result
                    expected[name + 'n'] = model.get_feature_names_out().reshape(1, -1)
                lines.extend(['@no_inline', f'def case{cases}() raises:'])
                matrix('X', X)
                matrix('T', T)
                lines.append(f'    exercise_indicator("{name}", X, T, {string(marker)}, {string(features)}, {error_on_new})')
                cases += 1
                indicator_cases += 1
lines.append('def main() raises:')
lines.extend(f'    case{i}()' for i in range(cases))
source = builds / 'string_parity.mojo'
source.write_text('\n'.join(lines) + '\n')
check.compile(source, builds / 'string_parity')
result = check.run(builds / 'string_parity', capture_output=True, text=True)
(builds / 'string-parity-output.txt').write_text(result.stdout)
output = iter(result.stdout.splitlines())
seen = set()
for line in output:
    header = line.split()
    kind, name = header[:2]
    assert name in expected and name not in seen, (name, line)
    seen.add(name)
    wanted = expected[name]
    if kind == 'E':
        assert wanted is None, name
        continue
    assert wanted is not None, name
    rows, columns = map(int, header[2:])
    got = []
    for _ in range(rows * columns):
        if kind == 'B':
            got.append(bool(int(next(output))))
        else:
            cell = next(output).split()
            if cell[0] == 'N':
                got.append(None)
            else:
                assert cell[0] == 'V', cell
                got.append(''.join(chr(int(next(output))) for _ in range(int(cell[1]))))
    actual = np.array(got, dtype=bool if kind == 'B' else object).reshape(rows, columns)
    np.testing.assert_array_equal(actual, wanted, err_msg=name)
assert seen == set(expected), set(expected) - seen
report = dict(cases=cases, standalone_indicator_cases=indicator_cases, records=len(seen), seed=317, comparison='exact Unicode strings and Boolean values', native_limits='retained empty modes reject numeric zero; inverse unsupported; no numeric parsing or heterogeneous categories')
evidence = ROOT / 'docs/private/impute.simple/strings'
evidence.mkdir(parents=True, exist_ok=True)
(evidence / 'parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS: {cases} string sklearn cases and native-limit checks, {len(seen)} records')
