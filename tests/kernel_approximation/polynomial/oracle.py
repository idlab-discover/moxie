"""Execute the pinned local sklearn PolynomialCountSketch methods with scipy.fft.

NumPy/SciPy supply validation plumbing and the FFT exactly as the pinned
module imports them (`from scipy.fft import fft, ifft`).
"""
import ast, subprocess
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft
ROOT = Path(__file__).resolve().parents[3]
REF = ROOT.parent / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
METHODS = ['__init__', 'fit', 'transform']


def load():
    assert subprocess.check_output(['git', '-C', str(REF), 'rev-parse', 'HEAD'], text=True).strip() == REV
    paths = ['sklearn/kernel_approximation.py']
    assert not subprocess.check_output(['git', '-C', str(REF), 'status', '--porcelain', '--', *paths], text=True).strip()
    source = REF / paths[0]

    def validate_data(est, X, reset=True, **kwargs):
        X = np.array(X, copy=True)
        if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
            raise ValueError('invalid matrix')
        if reset:
            est.n_features_in_ = X.shape[1]
        elif X.shape[1] != est.n_features_in_:
            raise ValueError('features')
        return X

    def check_random_state(seed):
        return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    namespace = {'np': np, 'sp': sp, 'fft': fft, 'ifft': ifft, 'validate_data': validate_data,
                 'check_random_state': check_random_state, 'check_is_fitted': lambda est: None}
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == 'PolynomialCountSketch':
            node.bases = []; node.decorator_list = []
            node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in METHODS]
            for method in node.body:
                method.decorator_list = []
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    cls = namespace['PolynomialCountSketch']
    assert cls.transform.__code__.co_filename == str(source)
    return cls
