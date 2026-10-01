"""Real Compose/FeatureUnion/Pipeline regression against the local sklearn source."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import compose_parity_io as io
from sklearn.base import clone

SOURCES = ['sklearn/pipeline.py', 'sklearn/compose/_column_transformer.py',
           'sklearn/linear_model/_base.py']
hashes = io.verify_reference(SOURCES)
pipeline = io.load_pinned(SOURCES[0])
ColumnTransformer = io.load_pinned(SOURCES[1]).ColumnTransformer
LinearRegression = io.load_pinned(SOURCES[2]).LinearRegression
for method, source in [(pipeline.Pipeline.fit, SOURCES[0]),
                       (ColumnTransformer.fit_transform, SOURCES[1]),
                       (LinearRegression.fit, SOURCES[2])]:
    io.pinned_code(method, source)
native = io.run(ROOT, ROOT / 'builds/tests/pipeline/test_compose_integration', capture=True).stdout
actual = io.parse_results(native)
assert set(actual) == {'0', '1'}
for ident, dtype, tol in [('0', np.float64, 1e-10), ('1', np.float32, 2e-5)]:
    i = np.arange(12)
    X = np.column_stack((i, i % 3, i % 2)).astype(dtype)
    y = (3 + 2 * i - 4 * (i % 3) + 5 * (i % 2)).astype(dtype)
    left = ColumnTransformer([('selected', 'passthrough', [2, 0])])
    right = ColumnTransformer([('selected', 'passthrough', [1])])
    union = pipeline.FeatureUnion([('left', left), ('right', right)],
                                  transformer_weights={'right': 0.5})
    pipe = pipeline.Pipeline([('union', union), ('model', LinearRegression())]).fit(X, y)
    assert pipe.n_features_in_ == 3
    assert pipe['union'].get_feature_names_out().tolist() == [
        'left__selected__x2', 'left__selected__x0', 'right__selected__x1']
    assert pipe['union'].get_feature_names_out(['a', 'b', 'c']).tolist() == [
        'left__selected__c', 'left__selected__a', 'right__selected__b']
    try:
        pipe['union'].get_feature_names_out([])
    except ValueError:
        pass
    else:
        raise AssertionError('explicit empty feature names must fail')
    outer = ColumnTransformer([('nested', clone(left), [0, 1, 2])]).fit(X)
    assert outer.get_feature_names_out()[0] == 'nested__selected__x2'
    expected = dict(transformed=union.transform(X), prediction=pipe.predict(X),
                    score=pipe.score(X, y), features=pipe.n_features_in_)
    fresh = clone(pipe).fit(X, y + 7)
    expected.update(clone_refit=fresh.predict(X), original=pipe.predict(X))
    pipe.fit(X, y + 7)
    expected['refit'] = pipe.predict(X)
    assert set(actual[ident]) == set(expected)
    for key, value in expected.items():
        exact = key in ('transformed', 'features')
        io.close(actual[ident][key], value, 0 if exact else tol, 0 if exact else tol, ident + ' ' + key)
folder = ROOT / 'builds/tests/pipeline/compose-parity'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'native.txt').write_text(native)
(folder / 'report.json').write_text(json.dumps(dict(
    revision=io.REVISION, sources=hashes, cases=2,
    tolerances={'float32': 2e-5, 'float64': 1e-10},
    numpy=np.__version__, support_sklearn=__import__('sklearn').__version__), indent=2) + '\n')
print('PASS: Compose/FeatureUnion/Pipeline/LinearRegression: two dtype parity cases')
