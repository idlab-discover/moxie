"""Compare keyword-update refits with the pinned local sklearn sources.

Float64 uses 2e-8 relative/absolute tolerance; Float32 uses 5e-5 for
accumulation/storage differences. Support masks and parameter values are exact.
No tie normalization is used for this deterministic dataset. The reference
graphical solvers reject Float32 assume_centered input in their compiled Gram
engine; those two refits use Float64 inputs rounded first to Float32, as in
the existing graphical parity suites.
"""
from pathlib import Path
import ast
import json
import sys
import warnings
import numpy as np
import sklearn.base

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io
import pinned_sklearn


def main():
    modules = pinned_sklearn.load()
    # Exercise get/set/clone from the same checkout as the covariance methods.
    path = pinned_sklearn.REFERENCE / 'sklearn/base.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseEstimator')
    names = {'get_params', 'set_params', '__sklearn_clone__'}
    nodes = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'clone', '_clone_parametrized'}]
    namespace = vars(sklearn.base).copy()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    out = ROOT / 'builds/params_parity'
    out.mkdir(parents=True, exist_ok=True)
    native = io.run(ROOT / 'builds/test_params', out / 'input.txt', '')
    classes = [modules['_robust_covariance'].MinCovDet, modules['_elliptic_envelope'].EllipticEnvelope,
               modules['_graph_lasso'].GraphicalLasso, modules['_graph_lasso'].GraphicalLassoCV]
    checks = 0
    warnings.simplefilter('ignore')
    for cls in classes:
        for name in names:
            setattr(cls, name, namespace[name])
            assert pinned_sklearn.executed_from_pinned(getattr(cls, name))
        robust = cls.__name__ in {'MinCovDet', 'EllipticEnvelope'}
        for dtype in (np.float64, np.float32):
            def data(features):
                return np.array([[(i * (j + 3) % 17) / 7.0 + i % (j + 2)
                                  for j in range(features)] for i in range(40)], dtype=dtype)
            model = cls(**({'random_state': 7} if robust else {})).fit(data(2))
            original = model.get_params()
            learned = {k: v.copy() if hasattr(v, 'copy') else v for k, v in vars(model).items() if k.endswith('_')}
            assert model.set_params() is model
            model.set_params(assume_centered=True)
            assert model.get_params() == dict(original, assume_centered=True)
            nullable = ('support_fraction', 'random_state') if robust else (('cv', 'n_jobs') if cls.__name__ == 'GraphicalLassoCV' else ('covariance',))
            updates = {'support_fraction': .8, 'random_state': 13} if robust else ({'cv': 3, 'n_jobs': 1} if cls.__name__ == 'GraphicalLassoCV' else {'covariance': 'precomputed'})
            model.set_params(**updates)
            model.set_params(**{key: None for key in nullable})
            assert all(model.get_params()[key] is None for key in nullable)
            for key, value in learned.items():
                if isinstance(value, dict):
                    for column in value:
                        np.testing.assert_array_equal(getattr(model, key)[column], value[column])
                else:
                    np.testing.assert_array_equal(getattr(model, key), value)
            copied = namespace['clone'](model)
            assert copied.get_params() == model.get_params()
            assert not hasattr(copied, 'covariance_')
            model.set_params(**original)
            updates = dict(assume_centered=True)
            if robust:
                updates.update(support_fraction=None, random_state=7)
            else:
                updates.update(verbose=False)
            if cls.__name__ == 'EllipticEnvelope':
                updates['contamination'] = .2
            model.set_params(**updates)
            refit_data = data(3)
            if not robust and dtype == np.float32:
                refit_data = refit_data.astype(np.float64)
            model.fit(refit_data)
            expected = namespace['clone'](model).fit(refit_data)
            np.testing.assert_array_equal(model.covariance_, expected.covariance_)
            prefix = f'{cls.__name__}_{dtype.__name__}'
            for attr in ('covariance', 'location', *(['support'] if robust else []),
                         *(['offset'] if cls.__name__ == 'EllipticEnvelope' else []),
                         *(['alpha'] if cls.__name__ == 'GraphicalLassoCV' else [])):
                actual = native[f'{prefix}_{attr}']
                reference = np.asarray(getattr(model, attr + '_')).ravel()
                if attr == 'support':
                    np.testing.assert_array_equal(actual, reference)
                else:
                    tol = 2e-8 if dtype == np.float64 else 5e-5
                    np.testing.assert_allclose(actual, reference, rtol=tol, atol=tol, err_msg=f'{prefix}_{attr}')
                checks += 1
    (out / 'summary.json').write_text(json.dumps(dict(cases=8, checks=checks, widened_graphical_refits=2, reference=pinned_sklearn.REVISION), indent=2) + '\n')
    print(f'PASS: keyword parameter parity, 8 cases, {checks} numerical comparisons')


if __name__ == '__main__':
    main()
