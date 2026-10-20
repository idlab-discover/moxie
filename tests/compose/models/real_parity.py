"""ColumnTransformer over real Moxie transformers vs pinned sklearn.

`ColumnTransformer` and `make_column_transformer` come from the pinned
`_column_transformer.py`; SimpleImputer, StandardScaler and MinMaxScaler come
from the pinned impute/preprocessing sources. The scenario mirrors
`real_components.mojo` and `real_positional.mojo`, which also assert the feature names checked here.

Tolerances: imputation and selection are exact. StandardScaler/MinMaxScaler
use Float64-accumulated moments in Moxie and NumPy reductions in sklearn, so
Float64 outputs match to 1e-12 and Float32 outputs to 2e-6 (a few Float32 ulps
of values of order 1). Median statistics are exact in both dtypes.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/models"
sys.path.insert(0, str(ROOT / 'tools'))
import compose_parity_io as io
from compose_parity_io import run

SOURCES = ['sklearn/compose/_column_transformer.py', 'sklearn/impute/_base.py',
           'sklearn/preprocessing/_data.py']
hashes = io.verify_reference(SOURCES)
module = io.load_pinned(SOURCES[0])
ColumnTransformer = module.ColumnTransformer
make_column_transformer = module.make_column_transformer
SimpleImputer = io.load_pinned(SOURCES[1]).SimpleImputer
data_module = io.load_pinned(SOURCES[2])
StandardScaler = data_module.StandardScaler
MinMaxScaler = data_module.MinMaxScaler
for method, source in [(ColumnTransformer.fit_transform, SOURCES[0]),
                       (SimpleImputer.fit, SOURCES[1]), (StandardScaler.fit, SOURCES[2])]:
    io.pinned_code(method, source)

NAMES = ['age', 'fare', 'sibsp', 'pclass']


def data(rows, offset, dtype):
    k = np.arange(rows) + offset
    X = np.column_stack([20 + (k * 7) % 31, ((k * 13) % 17) * 3.25 + 0.5, k % 4, 1 + k % 3]).astype(dtype)
    X[k % 5 == 1, 0] = np.nan
    X[k % 7 == 3, 1] = np.nan
    return X


def expect_names(actual, expected, label):
    assert list(actual) == expected, f'{label}: {list(actual)} != {expected}'


output = ''.join(run(ROOT, BUILD / name, capture=True).stdout
                 for name in ['real_components', 'real_positional'])
actual = io.parse_results(output)
assert set(actual) == {'0-named', '1-named', '0-positional', '1-positional'}, actual
POSITIONAL = {'positional', 'positional_transform', 'auto'}
report = []
for bits, dtype in [(64, np.float64), (32, np.float32)]:
    X, Z = data(12, 0, dtype), data(5, 12, dtype)
    frame, zframe = pd.DataFrame(X, columns=NAMES), pd.DataFrame(Z, columns=NAMES)
    named = ColumnTransformer([('num', SimpleImputer(strategy='median'), ['age', 'fare']),
                               ('cnt', StandardScaler(), ['sibsp'])], remainder=MinMaxScaler())
    expected = {'fit_transform': named.fit_transform(frame), 'transform': named.transform(zframe),
                'statistics': named.named_transformers_['num'].statistics_}
    expect_names(named.get_feature_names_out(), ['num__age', 'num__fare', 'cnt__sibsp', 'remainder__pclass'], 'named')
    assert named.output_indices_['num'] == slice(0, 2) and named.output_indices_['remainder'] == slice(3, 4)
    positional = ColumnTransformer([('num', SimpleImputer(strategy='median'), [0, 1]),
                                    ('cnt', StandardScaler(), [2])], remainder='passthrough')
    expected['positional'] = positional.fit_transform(X)
    expected['positional_transform'] = positional.transform(Z)
    expect_names(positional.get_feature_names_out(), ['num__x0', 'num__x1', 'cnt__x2', 'remainder__x3'], 'positional')
    short = ColumnTransformer([('num', SimpleImputer(strategy='median'), [0, 1]),
                               ('cnt', StandardScaler(), [2])], remainder='passthrough',
                              verbose_feature_names_out=False).fit(frame)
    expect_names(short.get_feature_names_out(), NAMES, 'short')
    auto = make_column_transformer((SimpleImputer(strategy='median'), [0, 1]), (StandardScaler(), [2]))
    expected['auto'] = auto.fit_transform(X)
    expect_names(auto.get_feature_names_out(), ['simpleimputer__x0', 'simpleimputer__x1', 'standardscaler__x2'], 'auto')
    named.fit(zframe)
    expected['refit_transform'] = named.transform(frame)
    tol = 2e-6 if bits == 32 else 1e-12
    errors = {}
    for part in ['named', 'positional']:
        ident = f'{int(bits == 32)}-{part}'
        got = actual[ident]
        wanted = {k: v for k, v in expected.items() if (k in POSITIONAL) == (part == 'positional')}
        assert set(got) == set(wanted), (sorted(got), sorted(wanted))
        errors.update({key: io.close(got[key], value, tol, tol, f'{ident} {key}') for key, value in wanted.items()})
    report.append(dict(bits=bits, errors=errors))

folder = BUILD / 'real_parity'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'native.txt').write_text(output)
(folder / 'report.json').write_text(json.dumps(dict(
    revision=io.REVISION, sources=hashes, numpy=np.__version__,
    support_sklearn=__import__('sklearn').__version__, results=report), indent=2) + '\n')
print('PASS: ColumnTransformer(SimpleImputer, StandardScaler, MinMaxScaler remainder) parity: '
      'Float32/Float64, named and positional keys, NaN imputation, feature names, refit')
