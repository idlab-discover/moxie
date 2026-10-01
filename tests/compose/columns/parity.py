"""ColumnTransformer parity against the pinned scikit-learn checkout.

`ColumnTransformer` and `make_column_transformer` are compiled from the pinned
`sklearn/compose/_column_transformer.py`; the support wheel supplies the
utilities it imports (indexing, validation, FunctionTransformer,
StandardScaler). Each scenario below is built identically by
`tests/compose/columns/parity_driver.mojo` with native test doubles:

- Scale(f)  <-> FunctionTransformer(lambda X: X * f, feature_names_out='one-to-one')
- Center    <-> StandardScaler(with_std=False)
- Expand    <-> FunctionTransformer interleaving [x, x**2] (names f, f^2)
- Short     <-> FunctionTransformer(lambda X: X[:-1])

Compared per scenario and dtype: fit_transform output, transform of unseen
rows, get_feature_names_out, output_indices_, transformers_ (name, fitted
kind, resolved columns) and, for error scenarios, the exception type and
message. make_column_transformer names are compared after mapping the
sklearn classes to the doubles' names (functiontransformer->scale,
standardscaler->center).

Tolerance: composition itself is exact (column selection, weights in the
input dtype, concatenation), so Scale/Expand/passthrough outputs must match
to rounding (1e-12 relative for Float64, 1e-6 for Float32, covering the
Float64-accumulated means of Center against NumPy's pairwise float64 mean).
"""
from pathlib import Path
from functools import partial
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/columns"
sys.path.insert(0, str(ROOT / 'tools'))
import compose_parity_io as io
from compose_parity_io import environment

SOURCES = ['sklearn/compose/_column_transformer.py', 'sklearn/base.py', 'sklearn/utils/_indexing.py']
hashes = io.verify_reference(SOURCES)
module = io.load_pinned('sklearn/compose/_column_transformer.py')
ColumnTransformer = module.ColumnTransformer
make_column_transformer = module.make_column_transformer
io.pinned_code(ColumnTransformer.fit_transform, SOURCES[0])
io.pinned_code(ColumnTransformer.transform, SOURCES[0])
io.pinned_code(ColumnTransformer.get_feature_names_out, SOURCES[0])
io.pinned_code(make_column_transformer, SOURCES[0])

from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _mul(X, f):
    return X * X.dtype.type(f)


def scale(f=1.0):
    return FunctionTransformer(partial(_mul, f=f), feature_names_out='one-to-one')


def center():
    return StandardScaler(with_std=False)


def _interleave(X):
    return np.stack([X, X * X], axis=2).reshape(X.shape[0], -1)


def _expand_names(transformer, names):
    return np.asarray([n + s for n in names for s in ('', '^2')], dtype=object)


def expand():
    return FunctionTransformer(_interleave, feature_names_out=_expand_names)


def short():
    return FunctionTransformer(lambda X: X[:-1])


def last_two(X):
    return [X.shape[1] - 2, X.shape[1] - 1]


F = False
T = True
SCENARIOS = {
    'lists': lambda: ColumnTransformer([('a', scale(2.0), [0, 1]), ('b', scale(-1.0), [3])]),
    'passthrough_weights': lambda: ColumnTransformer([('a', scale(2.0), [1])], remainder='passthrough',
                                                     transformer_weights={'a': 0.5, 'remainder': 10.0}),
    'remainder_estimator': lambda: ColumnTransformer([('a', 'drop', [0, 1])], remainder=center()),
    'forms': lambda: ColumnTransformer([('neg', 'passthrough', [-1, 0, 0]), ('sl', 'passthrough', slice(4, 0, -2)),
                                        ('mask', 'passthrough', [F, T, F, F, T])]),
    'callable': lambda: ColumnTransformer([('f', 'passthrough', last_two), ('c', center(), [0])],
                                          remainder='passthrough'),
    'empty': lambda: ColumnTransformer([('e', scale(), []), ('m', scale(), [F, F, F, F, F]), ('d', 'drop', [2]),
                                        ('a', expand(), [0, 1])], remainder='passthrough'),
    'masks': lambda: ColumnTransformer([('m', 'passthrough', [T, F, F, F, F]), ('n', center(), [F, F, T, F, F])]),
    'names_false': lambda: ColumnTransformer([('a', scale(), [0]), ('b', expand(), [2])], remainder='passthrough',
                                             verbose_feature_names_out=False),
    'names_format': lambda: ColumnTransformer([('a', scale(), [0]), ('b', expand(), [2])], remainder='passthrough',
                                              verbose_feature_names_out='{feature_name}@{transformer_name}'),
    'make': lambda: make_column_transformer((scale(2.0), [0]), (scale(3.0), [1]), ('passthrough', [2]),
                                            (center(), [3])),
    'dense_center': lambda: ColumnTransformer([('a', center(), [0, 1, 2, 3, 4])]),
    'err_duplicate': lambda: ColumnTransformer([('a', scale(), [0]), ('a', scale(), [1])]),
    'err_reserved': lambda: ColumnTransformer([('remainder', scale(), [0])]),
    'err_dunder': lambda: ColumnTransformer([('a__b', scale(), [0])]),
    'err_mode': lambda: ColumnTransformer([('a', 'dorp', [0])]),
    'err_remainder': lambda: ColumnTransformer([('a', scale(), [0])], remainder='keep'),
    'err_threshold': lambda: ColumnTransformer([('a', scale(), [0])], sparse_threshold=1.5),
    'err_range': lambda: ColumnTransformer([('a', scale(), [5])]),
    'err_mask': lambda: ColumnTransformer([('a', scale(), [T, F])]),
    'err_full_slice': lambda: ColumnTransformer([('a', scale(), slice(None, None, 2))]),
    'err_string': lambda: ColumnTransformer([('a', scale(), 'age')]),
    'err_scalar_center': lambda: ColumnTransformer([('c', center(), 2)]),
    'err_scalar_passthrough': lambda: ColumnTransformer([('p', 'passthrough', 2)]),
    'err_rows': lambda: ColumnTransformer([('a', 'passthrough', [0]), ('r', short(), [1])]),
    'err_collision': lambda: ColumnTransformer([('a', scale(), [0, 1]), ('b', 'passthrough', [1, 0])],
                                               verbose_feature_names_out=False),
    'err_features': lambda: ColumnTransformer([('a', scale(), [0])]),
}
# Messages compared only up to this prefix: Python prints a set whose order
# depends on string hashing.
PREFIX = {'err_remainder': "InvalidParameterError: The 'remainder' parameter of ColumnTransformer must be a str among"}
RENAME = {'functiontransformer': 'scale', 'standardscaler': 'center'}

rng = np.random.default_rng(20261001)
X = rng.normal(size=(6, 5)) * 3 + 1
Z = rng.normal(size=(3, 5)) * 3 + 1
cases = []
for scenario in SCENARIOS:
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        Zc = Z[:, :3] if scenario == 'err_features' else Z
        cases.append(dict(id=f'{scenario}-{bits}', scenario=scenario, bits=bits, dtype=dtype,
                          X=X.astype(dtype), Z=Zc.astype(dtype)))

folder = BUILD / 'parity'
folder.mkdir(parents=True, exist_ok=True)
lines = [io.case_line(c['id'], c['bits'], dict(scenario=c['scenario']), dict(X=c['X'], Z=c['Z'])) for c in cases]
(folder / 'fixtures.txt').write_text('\n'.join(lines) + '\n')
output = subprocess.run([str(BUILD / 'parity_driver'), str(folder / 'fixtures.txt')], text=True,
                        capture_output=True, check=True, env=environment(ROOT)).stdout
(folder / 'native.txt').write_text(output)
native = io.parse_results(output)
text = {}
for line in output.splitlines():
    kind, _, rest = line.partition('|')
    if kind in ('names', 'spans', 'records'):
        ident, _, body = rest.partition('|')
        text.setdefault(ident, {})[kind] = body
assert set(native) == {c['id'] for c in cases}, 'native driver did not report every case'


def kind_of(config, trans, columns):
    if isinstance(config, str):
        return config
    return 'unfitted' if module._is_empty_column_selection(columns) else 'estimator'


def records(ct):
    configs = {name: trans for name, trans, _ in ct.transformers}
    configs['remainder'] = ct.remainder
    out = ''
    for name, trans, columns in ct.transformers_:
        out += f'{name}={kind_of(configs[name], trans, columns)}={columns!s};'
    return out


report = []
for case in cases:
    got = native[case['id']]
    ct = SCENARIOS[case['scenario']]()
    try:
        out = ct.fit_transform(case['X'])
        tz = ct.transform(case['Z'])
        names = list(ct.get_feature_names_out())
    except Exception as error:
        expected = f'{type(error).__name__}: {error}'.split('\n')[0]
        assert 'error' in got, (case['id'], 'native succeeded', expected)
        prefix = PREFIX.get(case['scenario'])
        if prefix:
            assert expected.startswith(prefix) and got['error'].startswith(prefix), (case['id'], got['error'], expected)
        else:
            assert expected.startswith(got['error']) or got['error'].startswith(expected), \
                (case['id'], got['error'], expected)
        report.append(dict(case=case['id'], error=got['error']))
        continue
    assert 'error' not in got, (case['id'], got)
    assert out.dtype == case['dtype'] and tz.dtype == case['dtype'], case['id']
    tol = 1e-12 if case['bits'] == 64 else 1e-6
    errors = dict(out=io.close(got['out'].reshape(out.shape) if out.shape[1] else np.zeros(out.shape), out, tol, tol,
                               case['id'] + ' out'),
                  tz=io.close(got['tz'].reshape(tz.shape) if tz.shape[1] else np.zeros(tz.shape), tz, tol, tol,
                              case['id'] + ' tz'))
    native_names = text[case['id']]['names'].split(',') if text[case['id']]['names'] else []
    if case['scenario'] == 'make':
        renamed = []
        for n in names:
            prefix, _, rest = n.partition('__')
            base, dash, index = prefix.partition('-')
            renamed.append(RENAME.get(base, base) + dash + index + '__' + rest)
        names = renamed
    assert native_names == names, (case['id'], native_names, names)
    spans = dict(item.split(':', 1) for item in text[case['id']]['spans'].split(';') if item)
    expected_spans = {k: f'{v.start}:{v.stop}' for k, v in ct.output_indices_.items()}
    if case['scenario'] == 'make':
        expected_spans = {RENAME.get(k.split('-')[0], k.split('-')[0]) + k[len(k.split('-')[0]):]: v
                          for k, v in expected_spans.items()}
    assert spans == expected_spans, (case['id'], spans, expected_spans)
    expected_records = records(ct)
    if case['scenario'] == 'make':
        for old, new in RENAME.items():
            expected_records = expected_records.replace(old, new)
    assert text[case['id']]['records'] == expected_records, (case['id'], text[case['id']]['records'], expected_records)
    report.append(dict(case=case['id'], **errors))

metadata = dict(revision=io.REVISION, sources=hashes, cases=len(cases), seed=20261001, numpy=np.__version__,
                support_sklearn=__import__('sklearn').__version__, results=report)
(folder / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
numeric = [r for r in report if 'error' not in r]
worst = {b: max(max(r['out'], r['tz']) for r in numeric if r['case'].endswith(str(b))) for b in (32, 64)}
print(f'ColumnTransformer parity passed: {len(cases)} cases ({len(report) - len(numeric)} error cases) vs pinned '
      f'sklearn {io.REVISION[:10]}; max abs error float64={worst[64]:.2e}, float32={worst[32]:.2e}')
