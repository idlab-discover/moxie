"""Compare the dense adapter and default CV curves with pinned sklearn."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn
import sklearn.model_selection._validation as validation
import sklearn.utils._metadata_requests as metadata
from sklearn.metrics import make_scorer, r2_score

ROOT = Path(__file__).resolve().parents[5]
ORACLE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SPLIT = 'sklearn/model_selection/_split.py'
DUMMY = 'sklearn/dummy.py'
VALIDATION = 'sklearn/model_selection/_validation.py'
METADATA = 'sklearn/utils/_metadata_requests.py'
FILES = [SPLIT, DUMMY, VALIDATION, METADATA]
assert sklearn.__version__ == '1.9.1'
assert subprocess.check_output(['git', '-C', ORACLE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', ORACLE, 'status', '--porcelain', '--', *FILES], text=True)


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, ORACLE / source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_functions(source, namespace, names=None):
    path = ORACLE / source
    functions = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names)
    ]
    for function in functions:
        function.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


split = load('sklearn.model_selection._moxie_split_reference', SPLIT)
dummy = load('sklearn._moxie_dummy_reference', DUMMY)
meta = vars(metadata).copy()
load_functions(METADATA, meta, {'_manual_routing'})
cv = vars(validation).copy()
cv.update(
    check_cv=split.check_cv, _manual_routing=meta['_manual_routing'],
    array_device=lambda x: None, move_to=lambda x, **kwargs: np.asarray(x),
)
load_functions(VALIDATION, cv)
rows = np.arange(12)
X = rows.astype(np.float32).reshape(-1, 1)
y = (rows % 5).astype(np.float32)
weights = (rows + 1).astype(np.float32)

expected = {'score': [], 'train_score': [], 'index': []}
for fold, (train, test) in enumerate(split.check_cv(None, y).split(X, y)):
    model = dummy.DummyRegressor().fit(X[train], y[train], sample_weight=weights[train])
    expected['score'].append([fold, model.score(X[test], y[test], sample_weight=weights[test])])
    expected['train_score'].append([fold, model.score(X[train], y[train], sample_weight=weights[train])])
    expected['index'].extend([fold, row] for row in test)
model = dummy.DummyRegressor().fit(X, y, sample_weight=weights)
expected['prediction'] = [[model.predict(X)[0]]]

validation_train, validation_test = cv['validation_curve'](
    dummy.DummyRegressor(), X, y, param_name='strategy', param_range=['mean', 'median'],
)
with sklearn.config_context(enable_metadata_routing=True), warnings.catch_warnings():
    # R² is undefined for the smallest training prefix, with one sample.
    warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
    model = dummy.DummyRegressor().set_fit_request(sample_weight=True)
    scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
    sizes, learning_train, learning_test = cv['learning_curve'](
        model, X, y, scoring=scorer, params={'sample_weight': weights},
    )
expected['learning_size'] = sizes.reshape(-1, 1)
for name, train, test in (
    ('validation_curve', validation_train, validation_test),
    ('learning_curve', learning_train, learning_test),
):
    expected[name] = [[i, j, train[i, j], test[i, j]] for i, j in np.ndindex(train.shape)]

actual = {}
for line in subprocess.check_output([sys.argv[1]], text=True).splitlines():
    key, *values = line.split()
    actual.setdefault(key, []).append([float(value) for value in values])
assert actual.keys() == expected.keys()
for key, values in expected.items():
    observed = np.asarray(actual[key])
    values = np.asarray(values)
    if key in ('index', 'learning_size'):
        np.testing.assert_array_equal(observed, values, err_msg=key)
        continue
    if key in ('validation_curve', 'learning_curve'):
        np.testing.assert_array_equal(observed[:, :2], values[:, :2], err_msg=key)
        observed, values = observed[:, 2:], values[:, 2:]
    # Float32 weighted means can round differently, especially in small prefixes.
    rtol = 2e-7 if key == 'learning_curve' else 0
    np.testing.assert_allclose(observed, values, rtol=rtol, atol=2e-6, err_msg=key)
print(f'PASS pinned dense adapter and CV curves with {REVISION}')
