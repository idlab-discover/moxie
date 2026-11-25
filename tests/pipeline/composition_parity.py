"""Compare Pipeline search and nested SVC search with pinned sklearn.

GridSearchCV and the validation functions come from the pinned checkout's model_selection
sources (loaded over the locked wheel); estimators come from the locked wheel.
"""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import sklearn
import sklearn.model_selection._validation as wheel_validation
import sklearn.utils._metadata_requests as metadata
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', SOURCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
FILES = [
    'sklearn/model_selection/_search.py',
    'sklearn/model_selection/_validation.py',
    'sklearn/model_selection/_split.py',
    'sklearn/utils/_metadata_requests.py',
    'sklearn/pipeline.py',
]
assert not subprocess.check_output(['git', '-C', SOURCE, 'status', '--porcelain', '--', *FILES], text=True)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split = load('sklearn.model_selection._moxie_split_reference', 'sklearn/model_selection/_split.py')
search = load('sklearn.model_selection._moxie_search_reference', 'sklearn/model_selection/_search.py')
# Same wiring as families/model_selection/algorithms/search/tests/search_parity.py:
# pinned CV bodies over the 1.9.1 wheel's validation, callbacks and tags.
ns = vars(wheel_validation).copy()
ns.update(array_device=lambda x: None, move_to=lambda x, **kwargs: np.asarray(x))
body = next(n for n in ast.parse((SOURCE / 'sklearn/utils/_metadata_requests.py').read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_manual_routing')
meta = vars(metadata).copy()
exec(compile(ast.Module(body=[body], type_ignores=[]), 'metadata', 'exec'), meta)
ns['_manual_routing'] = meta['_manual_routing']
functions = [n for n in ast.parse((SOURCE / 'sklearn/model_selection/_validation.py').read_text()).body
             if isinstance(n, ast.FunctionDef)]
for function in functions:
    function.decorator_list = []
exec(compile(ast.Module(body=functions, type_ignores=[]), 'validation', 'exec'), ns)
search.check_cv = ns['check_cv'] = split.check_cv
for name in ('_fit_and_score', '_aggregate_score_dicts', '_insert_error_scores',
             '_normalize_score_results', '_warn_or_raise_about_fit_failures'):
    setattr(search, name, ns[name])
GridSearchCV, cross_val_score = search.GridSearchCV, ns['cross_val_score']
Pipeline = load('sklearn._pipeline_pinned', 'sklearn/pipeline.py').Pipeline


def dataset():
    i = np.arange(30)
    X = np.column_stack([(i % 3) + i / 30, (i * 7) % 5, 10.0 * ((i * 11) % 4) - (i % 3)]).astype(float)
    return X, (i % 3).astype(float)


out = {}
assert sklearn.__version__ == '1.9.1'
X, y = dataset()
kf = KFold(3)
scaled = Pipeline([('standardscaler', StandardScaler()), ('svc', SVC())])
out['pipeline_cv'] = cross_val_score(scaled, X, y, cv=kf).tolist()
scaled_search = GridSearchCV(clone(scaled), {'svc__C': [0.1, 1.0, 10.0]}, cv=kf).fit(X, y)
pca_search = GridSearchCV(Pipeline([('pca', PCA()), ('svc', SVC())]),
                 {'pca__n_components': [1, 2], 'svc__C': [0.1, 10.0]}, cv=kf).fit(X, y)
scaled_pca = Pipeline([('standardscaler', StandardScaler()), ('pca', PCA()), ('svc', SVC())])
scaled_pca_search = GridSearchCV(scaled_pca,
    {'pca__n_components': [1, 2], 'svc__C': [0.1, 10.0]}).fit(X, y)
validation = ns['cross_validate'](
    scaled_pca, X, y, return_train_score=True, return_estimator=True, return_indices=True,
)
out['nested_pca_cv'] = validation['test_score']
out['nested_pca_train'] = validation['train_score']
out['nested_pca_fitted_score'] = [estimator.score(X, y) for estimator in validation['estimator']]
out['nested_pca_test_index'] = [
    (fold, row) for fold, test in enumerate(validation['indices']['test']) for row in test
]
inner = GridSearchCV(SVC(), {'C': [0.1, 1.0, 10.0]}, cv=KFold(3))
out['nested_cv'] = cross_val_score(inner, X, y, cv=kf).tolist()
nested_best = []
for train, test in kf.split(X):
    fitted = clone(inner).fit(X[train], y[train])
    nested_best.append(fitted.best_params_['C'])
out['nested_best_C'] = nested_best
final = Pipeline([('standardscaler', StandardScaler()), ('gridsearchcv', GridSearchCV(SVC(), {'C': [0.1, 1.0, 10.0]}, cv=KFold(3)))]).fit(X, y)
out['search_final_prediction'] = final.predict(X)
out['search_final_score'] = final.score(X, y)


def records(binary):
    output = subprocess.check_output([str(binary)], text=True)
    result = {}
    for line in output.splitlines():
        name, _, values = line.partition(" ")
        if name in ("PASS", "PASS:"):
            continue
        result.setdefault(name, []).append([float(v) for v in values.split()])
    return result


def compare(native, key, expected, *, atol=0):
    actual = np.asarray(native[key]).reshape(np.asarray(expected).shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=key)


def search_checks(native, prefix, fitted):
    result = fitted.cv_results_
    params = result['params']
    keys = sorted(params[0])
    compare(native, prefix + '_candidate', [[p[k] for k in keys] for p in params])
    compare(native, prefix + '_mean', result['mean_test_score'], atol=1e-12)
    compare(native, prefix + '_rank', result['rank_test_score'])
    compare(native, prefix + '_fold', np.stack([result[f'split{k}_test_score'] for k in range(fitted.n_splits_)], axis=1))
    compare(native, prefix + '_best', [fitted.best_index_])
    compare(native, prefix + '_prediction', fitted.predict(dataset()[0]))


binary_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'builds/search-composition'
scaled_native = records(binary_dir / 'pipeline_model_selection')
compare(scaled_native, 'pipeline_cv', out['pipeline_cv'])
search_checks(scaled_native, 'pipeline', scaled_search)
search_checks(scaled_native, 'pca', pca_search)
nested_pca_native = records(binary_dir / 'nested_pca')
compare(nested_pca_native, 'nested_pca_cv', out['nested_pca_cv'])
compare(nested_pca_native, 'nested_pca_train', out['nested_pca_train'])
compare(nested_pca_native, 'nested_pca_fitted_score', out['nested_pca_fitted_score'], atol=1e-15)
compare(nested_pca_native, 'nested_pca_test_index', out['nested_pca_test_index'])
search_checks(nested_pca_native, 'nested_pca', scaled_pca_search)
compare(nested_pca_native, 'nested_pca_weighted_score',
        scaled_pca_search.best_estimator_.score(X, y, sample_weight=np.arange(1, len(y) + 1)),
        atol=1e-12)
nested_native = records(binary_dir / 'nested_search')
compare(nested_native, 'nested_cv', out['nested_cv'])
compare(nested_native, 'nested_best', list(enumerate(out['nested_best_C'])))
compare(nested_native, 'search_final_prediction', out['search_final_prediction'])
compare(nested_native, 'search_final_score', out['search_final_score'])
print('PASS pinned composition parity: folds, candidate order, ties, ranks, best selection, and refit predictions')
print('Reference:', REVISION, 'with sklearn wheel', sklearn.__version__)
