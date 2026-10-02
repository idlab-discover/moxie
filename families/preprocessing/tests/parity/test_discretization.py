"""Parity: KBinsDiscretizer (uniform, every quantile method, weighted quantiles,
k-means; ordinal and onehot-dense encodings; subsampling).

The oracle runs single-threaded (OMP_NUM_THREADS=1) so sklearn's k-means
accumulates samples in order. Every native step reproduces the reference's
operations in the reference's precision, so n_bins_, bin_edges_, transforms,
inverse transforms and feature names are compared exactly (rtol=atol=0). See
docs/api/discretization.md for the stated exceptions (tie order of weighted
argsort and of k-means empty-cluster relocation), which these inputs avoid.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import warnings

import numpy as np

from harness import Driver, oracle

sk = oracle()


def _load_kmeans():
    """Import pinned sklearn.cluster._kmeans without sklearn/cluster/__init__.py.

    The package __init__ imports tree/linear_model/manifold modules whose pinned
    Python no longer matches the support wheel's compiled extensions. The
    cluster sources (Python and Cython) are byte-identical between the pinned
    revision and the wheel's 1.9.1 tag (checked below), so the wheel's k-means
    extensions execute the pinned algorithm.
    """
    import subprocess
    import sys
    import types
    from pathlib import Path
    reference = Path(__file__).resolve().parents[5] / 'scikit-learn'
    subprocess.run(['git', '-C', str(reference), 'diff', '--quiet', '1.9.1',
                    '4358542a1bd39af2e7e1cbace8b9fed067473c5b', '--', 'sklearn/cluster'], check=True)
    package = types.ModuleType('sklearn.cluster')
    package.__path__ = [str(Path(sk.__file__).parent / 'cluster')]
    sys.modules['sklearn.cluster'] = package
    from sklearn.cluster._kmeans import KMeans
    package.KMeans = KMeans


_load_kmeans()
from sklearn.preprocessing import KBinsDiscretizer

warnings.simplefilter('ignore')
rng = np.random.RandomState(0)
d = Driver('discretization')
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}
checks = []  # (case, name, expected, kind)

d.prelude('''
def fortran[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    """Column-major copy of x, exposed with x's logical shape (strides (1, n))."""
    var n = x.shape[0]
    var p = x.shape[1]
    var owner = zeros[d](Shape(n * p))
    for i in range(n):
        for j in range(p):
            owner.unsafe_set(j * n + i, x.item(i, j))
    return owner.view_with_layout(Shape(n, p), nm.NDArrayStrides(1, n), 0)

def reversed_rows[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    """Negative-stride view of x's rows in reverse order."""
    var n = x.shape[0]
    var p = x.shape[1]
    return x.view_with_layout(Shape(n, p), nm.NDArrayStrides(-p, 1), x.offset + (n - 1) * p)

def emit_model[T: DType, O: DType](prefix: String, est: KBinsDiscretizer[T, O]) raises:
    emit(prefix + "nb", est.n_bins_)
    for j in range(len(est.bin_edges_)):
        emit(prefix + "e" + String(j), est.bin_edges_[j])
''')


def expect(case, name, value, kind='exact'):
    checks.append((case, name, value, kind))


def expect_model(case, ref, prefix=''):
    expect(case, prefix + 'nb', ref.n_bins_)
    for j, edges in enumerate(ref.bin_edges_):
        expect(case, f'{prefix}e{j}', edges.astype(np.float64))


def mojo_params(nb, encode, strategy, method, subsample, seed):
    nb_text = f'List[Int]({list(nb)})' if isinstance(nb, list) else str(nb)
    sub = 'None' if subsample is None else str(subsample)
    rs = 'None' if seed is None else str(seed)
    return f'{nb_text}, encode="{encode}", strategy="{strategy}", quantile_method="{method}", subsample={sub}, random_state={rs}'


def add_case(tag, X, Xt, *, nb=4, encode='ordinal', strategy='quantile', method='averaged_inverted_cdf',
             weights=None, subsample=200_000, seed=None, out=None, layout=None, names=None):
    """One fitted model: learned attributes, transform of Xt, inverse, names."""
    dt = X.dtype.type
    ot = out or dt
    ref = KBinsDiscretizer(nb, encode=encode, strategy=strategy, quantile_method=method,
                           dtype=out, subsample=subsample, random_state=seed)
    fit_X = X if layout != 'fortran' else np.asfortranarray(X)
    fit_X = fit_X[::-1] if layout == 'reversed' else fit_X
    ref.fit(fit_X, sample_weight=None if weights is None else (weights if np.ndim(weights) == 0 else (weights[::-1] if layout == 'reversed' else weights)))
    view = {'fortran': 'fortran(X)', 'reversed': 'reversed_rows(X)', None: 'X'}[layout]
    body = [f'var X = {d.input(X)}', f'var Xt = {d.input(Xt)}',
            f'var est = KBinsDiscretizer[{MOJO[dt]}, {MOJO[ot]}]({mojo_params(nb, encode, strategy, method, subsample, seed)})']
    if weights is None:
        body.append(f'est.fit({view})' if names is None else f'est.fit_named({view}, {d.strings(names)})')
    elif np.ndim(weights) == 0:
        body.append(f'est.fit({view}, sample_weight={float(weights)})')
    else:
        W = d.input(np.asarray(weights[::-1] if layout == 'reversed' else weights, dtype=dt))
        body.append(f'est.fit({view}, sample_weight={W})')
    body += ['emit_model("", est)', 'var T = est.transform(Xt)', 'emit("t", T)',
             'emit("inv", est.inverse_transform(T))', 'emit_names("names", est.get_feature_names_out())',
             'emit("tview", est.transform(reversed_rows(Xt)))']
    expect_model(tag, ref)
    T = ref.transform(Xt)
    expect(tag, 't', T, 'typed')
    inv = ref.inverse_transform(T)
    if encode == 'onehot-dense':
        # Native inverse keeps the input array's dtype; sklearn returns Float64.
        inv = inv.astype(np.dtype(ot))
    expect(tag, 'inv', inv, 'typed')
    expect(tag, 'names', list(ref.get_feature_names_out(None if names is None else np.array(names, dtype=object))), 'names')
    expect(tag, 'tview', ref.transform(Xt[::-1]), 'typed')
    d.case(tag, '\n'.join(body))


# ---- datasets --------------------------------------------------------------
gauss = rng.randn(60, 3) * [1.0, 25.0, 1e-3] + [0.0, -40.0, 7.0]
skewed = np.column_stack([rng.exponential(2.0, 80), rng.lognormal(0, 1.5, 80), rng.randint(0, 6, 80).astype(float)])
offset = np.column_stack([1e6 + rng.randn(50) * 1e-3, rng.rand(50)])
constant = np.column_stack([np.full(20, 3.25), rng.randn(20), np.repeat([0.0, 1.0], 10)])
few = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [10.0, 2.0], [10.0, 3.0]])
tall = np.column_stack([rng.randn(700), rng.rand(700) * 100, rng.randint(0, 9, 700).astype(float)])
probe = lambda X: np.vstack([X[:7], X.min(0) - 1, X.max(0) + 1, (X.min(0) + X.max(0)) / 2])
datasets = {'gauss': gauss, 'skewed': skewed, 'offset': offset, 'constant': constant, 'few': few}

methods = ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf',
           'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased']
for dt in (np.float64, np.float32):
    s = dt.__name__
    for key, raw in datasets.items():
        X = raw.astype(dt)
        Xt = probe(X)
        nb = 3 if key == 'few' else 5
        add_case(f'uni_{key}_{s}', X, Xt, nb=nb, strategy='uniform')
        add_case(f'km_{key}_{s}', X, Xt, nb=nb, strategy='kmeans')
        for m in methods:
            add_case(f'q_{m}_{key}_{s}', X, Xt, nb=nb, method=m)
        w = (rng.rand(X.shape[0]) * 3).astype(dt)
        w[::7] = 0
        add_case(f'wq_avg_{key}_{s}', X, Xt, nb=nb, weights=w)
        add_case(f'wq_inv_{key}_{s}', X, Xt, nb=nb, method='inverted_cdf', weights=w)
        add_case(f'wu_{key}_{s}', X, Xt, nb=nb, strategy='uniform', weights=w)
        iw = rng.randint(1, 4, X.shape[0]).astype(dt)
        add_case(f'wqi_{key}_{s}', X, Xt, nb=nb, weights=iw)
        if key in ('gauss', 'offset'):
            # Weighted k-means only on tie-free columns: relocating an empty
            # cluster picks among equidistant samples in NumPy argpartition's
            # CPU-dependent order, which matters once their weights differ.
            add_case(f'wk_{key}_{s}', X, Xt, nb=nb, strategy='kmeans', weights=w)
            add_case(f'wki_{key}_{s}', X, Xt, nb=nb, strategy='kmeans', weights=iw)
    X = gauss.astype(dt)
    Xt = probe(X)
    # Encodings, output dtypes, layouts, per-feature bins, names, scalar weights.
    for strategy in ('uniform', 'quantile', 'kmeans'):
        add_case(f'oh_{strategy}_{s}', X, Xt, nb=[2, 3, 6], encode='onehot-dense', strategy=strategy)
        other = np.float32 if dt == np.float64 else np.float64
        add_case(f'out_{strategy}_{s}', X, Xt, encode='ordinal', strategy=strategy, out=other)
        add_case(f'ohout_{strategy}_{s}', X, Xt, encode='onehot-dense', strategy=strategy, out=other)
        add_case(f'fo_{strategy}_{s}', X, Xt, strategy=strategy, layout='fortran')
        add_case(f'rv_{strategy}_{s}', X, Xt, strategy=strategy, layout='reversed', weights=(rng.rand(60) + 0.5).astype(dt))
        add_case(f'sc_{strategy}_{s}', X, Xt, strategy=strategy, weights=2.5)
    add_case(f'named_{s}', X, Xt, encode='onehot-dense', names=['alpha', 'beta', 'gamma'])
    add_case(f'named_ord_{s}', X, Xt, names=['alpha', 'beta', 'gamma'])
    add_case(f'neg_uniform_{s}', X, Xt, strategy='uniform', weights=np.where(np.arange(60) % 5 == 0, -1.0, 1.0).astype(dt))
    # Subsampling (with replacement; weighted rows drawn by weight).
    T = tall.astype(dt)
    Tt = probe(T)
    tw = (rng.rand(700) * 2).astype(dt)
    for strategy in ('uniform', 'quantile', 'kmeans'):
        for seed in (0, 1234):
            add_case(f'sub_{strategy}_{seed}_{s}', T, Tt, nb=6, strategy=strategy, subsample=250, seed=seed)
            add_case(f'subw_{strategy}_{seed}_{s}', T, Tt, nb=6, strategy=strategy, subsample=250, seed=seed, weights=tw)
        add_case(f'nosub_{strategy}_{s}', T, Tt, nb=6, strategy=strategy, subsample=None)
    # Weighted + subsample allows any quantile method (weights already consumed).
    add_case(f'subw_linear_{s}', T, Tt, nb=6, method='linear', subsample=300, seed=5, weights=tw)
    # k-means relocation of an empty cluster (duplicate-heavy column).
    dup = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [10.0], [10.0], [4.0]]).astype(dt)
    add_case(f'km_empty_{s}', dup, probe(dup), nb=4, strategy='kmeans')
    add_case(f'km_empty_w_{s}', dup, probe(dup), nb=4, strategy='kmeans', weights=np.array([1, 2, 1, 1, 1, 3, 1, 0.5], dtype=dt))

# ---- refit replaces state, repeat transforms, onehot inverse ------------------
X = gauss
Y = skewed[:, :2]
ref = KBinsDiscretizer(3, encode='onehot-dense', strategy='uniform').fit(X)
ref2 = KBinsDiscretizer(3, encode='onehot-dense', strategy='uniform').fit(Y)
d.case('refit', f'''var est = KBinsDiscretizer(3, encode="onehot-dense", strategy="uniform")
est.fit({d.input(X)})
est.fit({d.input(Y)})
emit_model("", est)
emit_int("nfeat", est.n_features_in_)
var Z = est.transform({d.input(Y)})
emit("t", Z)
emit("inv", est.inverse_transform(Z))''')
expect_model('refit', ref2)
expect('refit', 'nfeat', 2, 'int')
expect('refit', 't', ref2.transform(Y), 'typed')
expect('refit', 'inv', ref2.inverse_transform(ref2.transform(Y)), 'typed')

# A borrowed RandomState advances across fits, like a RandomState instance.
for dt in (np.float64, np.float32):
    T = tall.astype(dt)
    shared = np.random.RandomState(77)
    first = KBinsDiscretizer(5, encode='ordinal', subsample=200, random_state=shared).fit(T)
    first_edges = [e.copy() for e in first.bin_edges_]
    first.fit(T)
    tag = f'borrowed_{dt.__name__}'
    d.case(tag, f'''var T = {d.input(T)}
var rng = RandomState(77)
var est = KBinsDiscretizer[{MOJO[dt]}](5, encode="ordinal", subsample=200)
est.fit(T, rng)
emit_model("a", est)
est.fit(T, rng)
emit_model("b", est)''')
    for j, e in enumerate(first_edges):
        expect(tag, f'ae{j}', e.astype(np.float64))
    expect_model(tag, first, 'b')

# Ordinal inverse indexing: truncation and NumPy's negative wrap.
ordinal = KBinsDiscretizer(4, encode='ordinal', strategy='uniform').fit(X)
codes = np.array([[0.9, 3.7, -1.0], [2.2, -4.0, 1.0]])
expect('ord_index', 'inv', ordinal.inverse_transform(codes), 'typed')
d.case('ord_index', f'''var est = KBinsDiscretizer(4, encode="ordinal", strategy="uniform")
est.fit({d.input(X)})
emit("inv", est.inverse_transform({d.input(codes)}))''')

# ---- errors ------------------------------------------------------------------
errors = []  # (name, native statement, sklearn callable or None, needle)


def sk_raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return None


Xe = d.input(gauss)
nan_X = gauss.copy()
nan_X[3, 1] = np.nan
cases = [
    ('bad_encode', 'KBinsDiscretizer(3, encode="dense").fit(X)', lambda: KBinsDiscretizer(3, encode='dense').fit(gauss), "'encode' parameter"),
    ('bad_strategy', 'KBinsDiscretizer(3, encode="ordinal", strategy="median").fit(X)', lambda: KBinsDiscretizer(3, encode='ordinal', strategy='median').fit(gauss), "'strategy' parameter"),
    ('bad_method', 'KBinsDiscretizer(3, encode="ordinal", quantile_method="lower").fit(X)', lambda: KBinsDiscretizer(3, encode='ordinal', quantile_method='lower').fit(gauss), "'quantile_method' parameter"),
    ('one_bin', 'KBinsDiscretizer(1, encode="ordinal").fit(X)', lambda: KBinsDiscretizer(1, encode='ordinal').fit(gauss), "'n_bins' parameter"),
    ('subsample0', 'KBinsDiscretizer(3, encode="ordinal", subsample=0).fit(X)', lambda: KBinsDiscretizer(3, encode='ordinal', subsample=0).fit(gauss), "'subsample' parameter"),
    ('bad_seed', 'KBinsDiscretizer(3, encode="ordinal", random_state=-1).fit(X)', lambda: KBinsDiscretizer(3, encode='ordinal', random_state=-1).fit(gauss), "'random_state' parameter"),
    ('nbins_len', 'KBinsDiscretizer(List[Int]([3, 3]), encode="ordinal").fit(X)', lambda: KBinsDiscretizer([3, 3], encode='ordinal').fit(gauss), 'n_bins must be a scalar or array of shape (n_features,).'),
    ('nbins_small', 'KBinsDiscretizer(List[Int]([1, 3, 0]), encode="ordinal").fit(X)', lambda: KBinsDiscretizer([1, 3, 0], encode='ordinal').fit(gauss), 'invalid number of bins at indices 0, 2.'),
    ('weighted_linear', f'KBinsDiscretizer(3, encode="ordinal", quantile_method="linear").fit(X, sample_weight={d.input(np.ones(60))})', lambda: KBinsDiscretizer(3, encode='ordinal', quantile_method='linear').fit(gauss, sample_weight=np.ones(60)), "quantile_method should either be set to 'averaged_inverted_cdf' or 'inverted_cdf'"),
    ('kmeans_small', f'KBinsDiscretizer(5, encode="ordinal", strategy="kmeans").fit({d.input(gauss[:3])})', lambda: KBinsDiscretizer(5, encode='ordinal', strategy='kmeans').fit(gauss[:3]), 'n_samples=3 should be >= n_clusters=5.'),
    ('nan_fit', f'KBinsDiscretizer(3, encode="ordinal").fit({d.input(nan_X)})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(nan_X), 'NaN'),
    ('weight_shape', f'KBinsDiscretizer(3, encode="ordinal").fit(X, sample_weight={d.input(np.ones(5))})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss, sample_weight=np.ones(5)), 'sample_weight.shape == (5,), expected (60,)!'),
    ('weight_zero', f'KBinsDiscretizer(3, encode="ordinal").fit(X, sample_weight={d.input(np.zeros(60))})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss, sample_weight=np.zeros(60)), 'at least one non-zero'),
    ('weight_nan', f'KBinsDiscretizer(3, encode="ordinal").fit(X, sample_weight={d.input(np.where(np.arange(60) == 4, np.nan, 1.0))})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss, sample_weight=np.where(np.arange(60) == 4, np.nan, 1.0)), 'NaN'),
    ('negative_subsample', f'KBinsDiscretizer(3, encode="ordinal", subsample=10, random_state=0).fit(X, sample_weight={d.input(np.where(np.arange(60) == 4, -1.0, 1.0))})', lambda: KBinsDiscretizer(3, encode='ordinal', subsample=10, random_state=0).fit(gauss, sample_weight=np.where(np.arange(60) == 4, -1.0, 1.0)), 'probabilities'),
    ('unfitted', '_ = KBinsDiscretizer(3, encode="ordinal").transform(X)', lambda: KBinsDiscretizer(3, encode='ordinal').transform(gauss), 'NotFittedError'),
    ('unfitted_names', '_ = KBinsDiscretizer(3, encode="ordinal").get_feature_names_out()', lambda: KBinsDiscretizer(3, encode='ordinal').get_feature_names_out(), 'NotFittedError'),
    ('n_features', f'var e = KBinsDiscretizer(3, encode="ordinal")\ne.fit(X)\n_ = e.transform({d.input(gauss[:, :2])})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss).transform(gauss[:, :2]), 'X has 2 features, but KBinsDiscretizer is expecting 3 features as input.'),
    ('nan_transform', f'var e = KBinsDiscretizer(3, encode="ordinal")\ne.fit(X)\n_ = e.transform({d.input(nan_X)})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss).transform(nan_X), 'NaN'),
    ('inv_cols', f'var e = KBinsDiscretizer(3, encode="ordinal")\ne.fit(X)\n_ = e.inverse_transform({d.input(np.zeros((2, 2)))})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss).inverse_transform(np.zeros((2, 2))), 'Incorrect number of features. Expecting 3, received 2.'),
    ('inv_index', f'var e = KBinsDiscretizer(3, encode="ordinal")\ne.fit(X)\n_ = e.inverse_transform({d.input(np.array([[0.0, 3.0, 0.0]]))})', lambda: KBinsDiscretizer(3, encode='ordinal').fit(gauss).inverse_transform(np.array([[0.0, 3.0, 0.0]])), 'index 3 is out of bounds for axis 0 with size 3'),
    ('inv_oh_cols', f'var e = KBinsDiscretizer(3, encode="onehot-dense")\ne.fit(X)\n_ = e.inverse_transform({d.input(np.zeros((2, 3)))})', lambda: KBinsDiscretizer(3, encode='onehot-dense').fit(gauss).inverse_transform(np.zeros((2, 3))), 'Expected 9 columns, got 3.'),
    ('inv_oh_zero', f'var e = KBinsDiscretizer(3, encode="onehot-dense")\ne.fit(X)\nvar Z = e.transform(X)\nZ.unsafe_set(0, 0)\nZ.unsafe_set(1, 0)\nZ.unsafe_set(2, 0)\n_ = e.inverse_transform(Z)', None, 'Samples [0] can not be inverted'),
    ('names_mismatch', f'var e = KBinsDiscretizer(3, encode="ordinal")\ne.fit_named(X, {d.strings(["a", "b", "c"])})\n_ = e.get_feature_names_out({d.strings(["a", "b", "z"])})', None, 'input_features is not equal to feature_names_in_'),
    ('onehot_sparse', 'KBinsDiscretizer(3).fit(X)', None, "encode='onehot' returns a sparse matrix"),
]
# sklearn's all-zero onehot row error, checked on its own transformed matrix:
Z = KBinsDiscretizer(3, encode='onehot-dense').fit(gauss).transform(gauss)
Z[0, :3] = 0
for name, stmt, sk, needle in cases:
    if sk is not None:
        assert sk_raises(sk), f'sklearn did not raise for {name}'
body = [f'var X = {Xe}']
for name, stmt, sk, needle in cases:
    if stmt.startswith('KBinsDiscretizer(') and ').fit(' in stmt:
        ctor, _, args = stmt.rpartition(').fit(')
        stmt = f'var e = {ctor})\ne.fit({args}'
    body.append('try:')
    body += ['    ' + line for line in stmt.split('\n')]
    body += [f'    emit_int("{name}_noerror", 1)', 'except e:', f'    emit_error("{name}", e)']
d.case('errors', '\n'.join(body))
assert sk_raises(lambda: KBinsDiscretizer(3, encode='onehot-dense').fit(gauss).inverse_transform(Z))

out = d.run(imports='from moxie.preprocessing import KBinsDiscretizer\nfrom moxie._random import RandomState')
for case, name, expected, kind in checks:
    got = out.get(case, {})
    if 'error' in got:
        d.close(f'{case}.{name}', got['error'], np.zeros(1))
        continue
    value = got.get(name, 'missing')
    if kind == 'names':
        d.equal(f'{case}.{name}', value, expected)
    elif kind == 'int':
        d.equal(f'{case}.{name}', value, expected)
    elif kind == 'typed':
        d.close(f'{case}.{name}', value, expected, rtol=0, atol=0, dtype=np.asarray(expected).dtype)
    else:
        d.close(f'{case}.{name}', value, np.asarray(expected), rtol=0, atol=0)
for name, stmt, sk, needle in cases:
    d.raises(f'errors.{name}', out['errors'].get(name, 'no error'), needle)
d.finish()
