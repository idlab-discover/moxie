"""Composition parity against local sklearn Pipeline, ColumnTransformer and Dummy.

These small integer selections, means and medians are exactly representable in
both dtypes, so all numerical comparisons are exact. Failed column fits preserve
native learned state by documented policy; sklearn does not promise rollback.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT
BUILD = ROOT / "builds/tests/compose/pipeline"
sys.path.insert(0, str(FAMILY / 'tools'))
import compose_parity_io as io
from compose_parity_io import run
from sklearn.base import clone

SOURCES = ['sklearn/pipeline.py', 'sklearn/compose/_column_transformer.py', 'sklearn/dummy.py']
hashes = io.verify_reference(SOURCES)
Pipeline = io.load_pinned(SOURCES[0]).Pipeline
ColumnTransformer = io.load_pinned(SOURCES[1]).ColumnTransformer
DummyRegressor = io.load_pinned(SOURCES[2]).DummyRegressor
for method, source in [(Pipeline.fit, SOURCES[0]), (ColumnTransformer.fit_transform, SOURCES[1]),
                       (DummyRegressor.fit, SOURCES[2])]:
    io.pinned_code(method, source)

native = run(ROOT, BUILD / 'main', capture=True).stdout
actual = io.parse_results(native)
assert set(actual) == {'0-0', '0-1', '1-0', '1-1'}, actual
for bits, dtype in [(64, np.float64), (32, np.float32)]:
    for multi in [False, True]:
        X = (np.arange(4)[:, None] * 10 + np.arange(3)).astype(dtype)
        Z = (100 + np.arange(2)[:, None] * 10 + np.arange(3)).astype(dtype)
        y = (np.arange(4) * 2).astype(dtype)
        if multi:
            y = np.column_stack((y, y + 10))
        pipe = Pipeline([('columns', ColumnTransformer(
            [('selected', 'passthrough', [2, 0, 2])], remainder='passthrough', n_jobs=2)),
                         ('dummy', DummyRegressor())])
        pipe.fit(X, y)
        assert pipe['dummy'].n_features_in_ == 4
        expected = dict(transformed=pipe['columns'].transform(Z), mean=pipe.predict(Z))
        assert pipe['columns'].get_feature_names_out().tolist() == [
            'selected__x2', 'selected__x0', 'selected__x2', 'remainder__x1']
        fresh = clone(pipe)
        fresh.set_params(dummy__strategy='median')
        y.flat[0] = 100
        fresh.fit(X, y)
        expected.update(median=fresh.predict(Z), original=pipe.predict(Z))
        pipe.fit(X, y)
        expected['refit'] = pipe.predict(Z)
        expected['after_failure'] = expected['refit']
        pipe.set_params(columns__remainder='drop')
        assert pipe['columns'].transform(Z).shape[1] == 4
        pipe.fit(X, y)
        expected.update(drop_remainder=pipe['columns'].transform(Z), updated=pipe.predict(Z))
        assert pipe['dummy'].n_features_in_ == 3
        ident = f'{int(bits == 32)}-{int(multi)}'
        assert set(actual[ident]) == set(expected), actual[ident]
        for key, value in expected.items():
            io.close(actual[ident][key], value, 0, 0, ident + ' ' + key)

folder = BUILD / 'parity'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'native.txt').write_text(native)
(folder / 'report.json').write_text(json.dumps(dict(
    revision=io.REVISION, sources=hashes, cases=4, exact=True,
    numpy=np.__version__, support_sklearn=__import__('sklearn').__version__), indent=2) + '\n')
print('PASS: Pipeline(ColumnTransformer, Moxie DummyRegressor): four exact parity cases; '
      'Float32/Float64, vector/multioutput targets, clone, nested parameters and refit')
