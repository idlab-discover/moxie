"""Check the real composed consumer against pinned sklearn estimator sources.

Tolerances follow the adopted LinearRegression solver evidence: 1e-8 for
Float64 and 3e-4 for Float32 (native widened Jacobi vs sklearn LAPACK).
The Float32 R2 tolerance is 3e-3 because native scoring widens accumulation.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT
BUILD = ROOT / "builds/tests/compose/models"
sys.path.insert(0, str(FAMILY / 'tools'))
import compose_parity_io as io
from compose_parity_io import run

SOURCES = ['sklearn/compose/_column_transformer.py', 'sklearn/compose/_target.py',
           'sklearn/preprocessing/_function_transformer.py', 'sklearn/linear_model/_base.py']
hashes = io.verify_reference(SOURCES)
import sklearn.preprocessing
import sklearn.linear_model
function = io.load_pinned(SOURCES[2], register=True).FunctionTransformer
linear = io.load_pinned(SOURCES[3], register=True).LinearRegression
sklearn.preprocessing.FunctionTransformer = function
sklearn.linear_model.LinearRegression = linear
target = io.load_pinned(SOURCES[1]).TransformedTargetRegressor
columns = io.load_pinned(SOURCES[0]).ColumnTransformer
for method, source in [(columns.fit_transform, SOURCES[0]), (target.fit, SOURCES[1]),
                       (function.fit, SOURCES[2]), (linear.fit, SOURCES[3])]:
    io.pinned_code(method, source)

output = run(ROOT, BUILD / 'main', capture=True).stdout
actual = io.parse_results(output)
assert set(actual) == {'0-1', '0-2', '1-1', '1-2'}, actual
report = []
for bits, dtype in [(64, np.float64), (32, np.float32)]:
    i = np.arange(12)
    X = np.column_stack(((i + 1)**2, 100 + i, i % 3)).astype(dtype)
    y = ((3 + 2 * (i + 1) + i % 3)**2 + i % 2).astype(dtype)
    weights = (i % 4).astype(dtype)
    for jobs in [1, 2]:
        ct = columns([('root', function(np.sqrt, inverse_func=np.square, validate=True), [0]),
                      ('raw', 'passthrough', [2])], n_jobs=jobs)
        model = target(regressor=linear(),
                       transformer=function(np.sqrt, inverse_func=np.square, validate=True))
        transformed = ct.fit_transform(X, y)
        model.fit(transformed, y, sample_weight=weights)
        prediction = model.predict(ct.transform(X))
        expected = dict(transformed=transformed, prediction=prediction, after_failure=prediction,
                        score=model.score(transformed, y, sample_weight=weights))
        ct.fit(X, y * 4)
        model.fit(ct.transform(X), y * 4)
        expected['refit'] = model.predict(ct.transform(X))
        expected['clone_refit'] = expected['refit']
        ident = f'{int(bits == 32)}-{jobs}'
        got = actual[ident]
        assert set(got) == set(expected), got
        errors = {}
        for key, value in expected.items():
            tol = (3e-3 if key == 'score' else 3e-4) if bits == 32 else 1e-8
            errors[key] = io.close(got[key], value, tol, tol, ident + ' ' + key)
        report.append(dict(bits=bits, jobs=jobs, errors=errors))

folder = BUILD / 'parity'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'native.txt').write_text(output)
(folder / 'report.json').write_text(json.dumps(dict(
    revision=io.REVISION, sources=hashes, numpy=np.__version__,
    support_sklearn=__import__('sklearn').__version__, results=report), indent=2) + '\n')
print('PASS: real ColumnTransformer -> TransformedTargetRegressor(LinearRegression) '
      'consumer parity: Float32/Float64, serial/parallel, weighted fit and refit')
