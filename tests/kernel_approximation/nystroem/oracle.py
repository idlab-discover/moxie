"""Execute the pinned local sklearn Nystroem methods; a support wheel supplies infrastructure.

The pinned `fit`, `transform` and `_get_kernel_params` bodies run unedited.
Kernel evaluation comes from the pinned `sklearn/metrics/pairwise.py`: every
kernel function Nystroem reaches is executed from the pinned source file as
well (loaded into the support wheel's module namespace, which supplies
Cython helpers and validation plumbing). The support wheel's own copies are
not used for the functions under comparison.
"""
import ast, builtins, subprocess, types
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import sklearn.metrics.pairwise as support
from sklearn.utils._array_api import get_namespace_and_device, _find_matching_floating_dtype
from sklearn.utils._indexing import _safe_indexing
ROOT = Path(__file__).resolve().parents[3]
REF = ROOT.parent / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
PAIRWISE = ['linear_kernel', 'polynomial_kernel', 'sigmoid_kernel', 'rbf_kernel', 'laplacian_kernel',
            'cosine_similarity', 'additive_chi2_kernel', 'chi2_kernel', 'euclidean_distances',
            'manhattan_distances', 'pairwise_kernels', '_pairwise_callable', '_parallel_pairwise', '_transposed_dist_wrapper']


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), [n.name for n in nodes]
    for n in nodes:
        n.decorator_list = []  # drop validate_params wrappers; Nystroem validates its own params
    exec(builtins.compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    for name in names:
        assert namespace[name].__code__.co_filename == str(path)


def load():
    assert subprocess.check_output(['git', '-C', str(REF), 'rev-parse', 'HEAD'], text=True).strip() == REV
    paths = ['sklearn/kernel_approximation.py', 'sklearn/metrics/pairwise.py']
    assert not subprocess.check_output(['git', '-C', str(REF), 'status', '--porcelain', '--', *paths], text=True).strip()
    pairwise_ns = dict(vars(support))
    _load_functions(REF / paths[1], PAIRWISE, pairwise_ns)
    # Route the kernel table used by pairwise_kernels to the pinned functions.
    table = {name: pairwise_ns[fn.__name__] for name, fn in support.PAIRWISE_KERNEL_FUNCTIONS.items()}
    pairwise_ns['PAIRWISE_KERNEL_FUNCTIONS'] = table
    pairwise_ns['kernel_metrics'] = lambda: table

    def validate_data(est, X, reset=True, **kwargs):
        X = np.array(X, copy=True)
        if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
            raise ValueError('invalid matrix')
        if reset:
            est.n_features_in_ = X.shape[1]
        elif X.shape[1] != est.n_features_in_:
            raise ValueError('features')
        return X

    namespace = {'np': np, 'sp': sp, 'warnings': __import__('warnings'), 'validate_data': validate_data,
                 'check_random_state': lambda s: s if isinstance(s, np.random.RandomState) else np.random.RandomState(s),
                 'check_is_fitted': lambda est: None, 'get_namespace_and_device': get_namespace_and_device,
                 '_find_matching_floating_dtype': _find_matching_floating_dtype, '_safe_indexing': _safe_indexing,
                 '_find_floating_dtype_allow_sparse': support._find_floating_dtype_allow_sparse,
                 'pairwise_kernels': pairwise_ns['pairwise_kernels'], 'KERNEL_PARAMS': support.KERNEL_PARAMS}
    nodes = []
    source = REF / paths[0]
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == 'Nystroem':
            node.bases = []; node.decorator_list = []
            node.body = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in ('__init__', 'fit', 'transform', '_get_kernel_params')]
            for method in node.body:
                method.decorator_list = []
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    cls = namespace['Nystroem']
    assert cls.fit.__code__.co_filename == str(source)
    return cls, pairwise_ns
