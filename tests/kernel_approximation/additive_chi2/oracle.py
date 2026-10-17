"""Execute the pinned local sklearn AdditiveChi2Sampler methods; NumPy supplies validation plumbing only."""
import ast, subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
REF = ROOT.parent / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
METHODS = ['__init__', 'fit', 'transform', 'get_feature_names_out', '_transform_dense']


def load():
    assert subprocess.check_output(['git', '-C', str(REF), 'rev-parse', 'HEAD'], text=True).strip() == REV
    paths = ['sklearn/kernel_approximation.py']
    assert not subprocess.check_output(['git', '-C', str(REF), 'status', '--porcelain', '--', *paths], text=True).strip()
    source = REF / paths[0]

    def validate_data(est, X, reset=True, ensure_non_negative=False, **kwargs):
        X = np.array(X, copy=True)
        if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
            raise ValueError('invalid matrix')
        if ensure_non_negative and (X < 0).any():
            raise ValueError('Negative values in data passed to AdditiveChi2Sampler')
        if reset:
            est.n_features_in_ = X.shape[1]
        elif hasattr(est, 'n_features_in_') and X.shape[1] != est.n_features_in_:
            raise ValueError('features')
        return X

    def check_is_fitted(est, attributes=None):
        if attributes and not hasattr(est, attributes):
            raise ValueError('not fitted')

    def check_feature_names_in(est, input_features, generate_names=True):
        if input_features is None:
            return np.asarray([f'x{i}' for i in range(est.n_features_in_)], dtype=object)
        if len(input_features) != est.n_features_in_:
            raise ValueError('input_features length')
        return np.asarray(input_features, dtype=object)

    namespace = {'np': np, 'sp': type('sp', (), {'issparse': staticmethod(lambda x: False)}),
                 'validate_data': validate_data, 'check_is_fitted': check_is_fitted,
                 '_check_feature_names_in': check_feature_names_in}
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == 'AdditiveChi2Sampler':
            node.bases = []; node.decorator_list = []
            node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in METHODS]
            assert sorted(m.name for m in node.body) == sorted(METHODS)
            for method in node.body:  # keep @staticmethod, drop sklearn's _fit_context
                method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == 'staticmethod']
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    cls = namespace['AdditiveChi2Sampler']
    assert cls.transform.__code__.co_filename == str(source)
    return cls
