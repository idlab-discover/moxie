"""Parity: QuantileTransformer, PowerTransformer, quantile_transform, power_transform.

Tolerances (see docs/api/quantiles.md for the derivation):

* Quantile statistics (`quantiles_`, `references_`) are exact: the same NumPy
  linear-percentile arithmetic on the same (subsampled) rows.
* Quantile transforms: Float64 rtol 1e-12 / atol 1e-15; Float32 is compared
  after rounding both sides' Float64 path, rtol 2e-7 (one Float32 ulp) since
  every operation is replayed in sklearn's dtype. Normal outputs use the Cephes
  `ndtr`/`ndtri` SciPy calls, so they agree to the last bits as well.
* Lambdas come from SciPy's iterative minimizers (xtol 1.48e-8): atol 2e-7
  + rtol 2e-7 bounds two runs that each stop within the optimizer tolerance;
  outputs derived from lambdas use rtol/atol 1e-6 (Float64) and 2e-5 (Float32).
"""
import os
import warnings

import numpy as np

from harness import Driver, oracle

oracle()
from sklearn import preprocessing as P

rng = np.random.RandomState(7)
d = Driver('quantiles')
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}
QTOL = {np.float64: dict(rtol=1e-12, atol=1e-15), np.float32: dict(rtol=2.4e-7, atol=1e-12)}
LTOL = dict(rtol=2e-7, atol=2e-7)
PTOL = {np.float64: dict(rtol=1e-6, atol=1e-6), np.float32: dict(rtol=2e-5, atol=2e-5)}
EXACT = dict(rtol=0, atol=0)
checks = []  # (case, name, expected, tolerance dict or kind, dtype)
DEBUG = os.environ.get('QDEBUG')
worst = {}


def expect(case, name, value, tol, dtype=None):
    checks.append((case, name, value, tol, dtype))


d.prelude('''
from moxie._random import RandomState

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
    var n = x.shape[0]
    var p = x.shape[1]
    return x.view_with_layout(Shape(n, p), nm.NDArrayStrides(-p, 1), x.offset + (n - 1) * p)

def emit_n(name: String, value: Int):
    emit_int(name, value)
''')


# ---------------------------------------------------------------------------
# QuantileTransformer

qdata = {
    'gauss': rng.randn(60, 3) * [1.0, 100.0, 1e-3] + [0.0, 5.0, -2.0],
    'nan': np.where(rng.rand(40, 3) < 0.2, np.nan, rng.randn(40, 3)),
    'nan_last_col': np.column_stack([rng.randn(12), rng.rand(12), np.full(12, np.nan)]),
    'ties': np.column_stack([0.1 * np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7] * 10),
                             np.repeat([1.0, 2.0, 3.0], 60)]),
    'toy': np.array([[0, 2, 2.6], [25, 4, 4.1], [50, 6, 2.3], [75, 8, 9.5], [100, 10, 0.1]]),
    'bounds': np.array([[0.0, 0], [0, 0], [1, 0]]),
    'single': np.array([[3.5, -1.0]]),
    'tiny_gap': np.array([[0.0], [1e-8], [1.5], [2], [3], [3], [4]]),
    'tall': rng.standard_exponential((2500, 2)),
}
configs = [
    dict(n_quantiles=1000, subsample=10000),
    dict(n_quantiles=5, subsample=None),
    dict(n_quantiles=17, subsample=40, random_state=0),
    dict(n_quantiles=1, subsample=None),
]
for dt in (np.float64, np.float32):
    for key, raw in qdata.items():
        X = raw.astype(dt)
        lo, hi = np.nanmin(X), np.nanmax(X)
        span = (hi - lo) if hi > lo else 1.0
        extra = np.full(X.shape[1], lo - span)
        extra[-1] = hi + span
        nanrow = X[0].copy()
        nanrow[0] = np.nan
        T = np.vstack([X[: min(len(X), 8)], extra, nanrow]).astype(dt)
        U = rng.uniform(-0.1, 1.1, (9, X.shape[1])).astype(dt)
        U[0, 0] = 0.0
        U[1, -1] = 1.0
        U[2, 0] = np.nan
        Z = (rng.randn(9, X.shape[1]) * 3).astype(dt)
        Z[0, 0] = -9.0
        Z[1, -1] = 9.0
        Z[2, 0] = np.nan
        tag = f'qt_{key}_{dt.__name__}'
        body = [f'var X = {d.input(X)}', f'var T = {d.input(T)}', f'var U = {d.input(U)}', f'var Z = {d.input(Z)}']
        for c, cfg in enumerate(configs):
            if cfg.get('subsample') and cfg['n_quantiles'] > cfg['subsample']:
                continue
            if cfg.get('subsample') == 40 and key == 'tall':
                cfg = dict(cfg, subsample=700, n_quantiles=300)
            for dist in ('uniform', 'normal'):
                n = f'{c}{dist[0]}'
                args = [f'n_quantiles={cfg["n_quantiles"]}', f'output_distribution="{dist}"',
                        f'subsample={cfg["subsample"] if cfg["subsample"] else "None"}']
                if 'random_state' in cfg:
                    args.append(f'random_state={cfg["random_state"]}')
                body += [f'var q{n} = QuantileTransformer[{MOJO[dt]}]({", ".join(args)})',
                         f'emit("ft{n}", q{n}.fit_transform(X))',
                         f'emit("q{n}", q{n}.quantiles_)', f'emit("r{n}", q{n}.references_)',
                         f'emit_n("nq{n}", q{n}.n_quantiles_)',
                         f'emit("t{n}", q{n}.transform(T))',
                         f'emit("i{n}", q{n}.inverse_transform({"U" if dist == "uniform" else "Z"}))',
                         f'emit("ii{n}", q{n}.inverse_transform(q{n}.transform(X)))',
                         f'var w{n} = T.copy()', f'q{n}.transform_inplace(w{n})', f'emit("tw{n}", w{n})']
                ref = P.QuantileTransformer(output_distribution=dist, **cfg)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    expect(tag, f'ft{n}', ref.fit_transform(X), QTOL[dt], dt)
                expect(tag, f'q{n}', ref.quantiles_, EXACT, np.float64)
                expect(tag, f'r{n}', ref.references_, EXACT, np.float64)
                expect(tag, f'nq{n}', ref.n_quantiles_, 'int')
                expect(tag, f't{n}', ref.transform(T), QTOL[dt], dt)
                expect(tag, f'i{n}', ref.inverse_transform(U if dist == 'uniform' else Z), QTOL[dt], dt)
                expect(tag, f'ii{n}', ref.inverse_transform(ref.transform(X)), QTOL[dt], dt)
                expect(tag, f'tw{n}', ref.transform(T), QTOL[dt], dt)
        if X.shape[0] > 1:
            # Strided inputs: column-major and reversed-row views fit/transform identically.
            body += [f'var fo = QuantileTransformer[{MOJO[dt]}](n_quantiles=7)', 'emit("fo", fo.fit_transform(fortran(X)))',
                     'emit("foq", fo.quantiles_)',
                     f'var rv = QuantileTransformer[{MOJO[dt]}](n_quantiles=7, output_distribution="normal")',
                     'emit("rv", rv.fit_transform(reversed_rows(X)))']
            ref = P.QuantileTransformer(n_quantiles=7)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                expect(tag, 'fo', ref.fit_transform(np.asfortranarray(X)), QTOL[dt], dt)
                expect(tag, 'foq', ref.quantiles_, EXACT, np.float64)
                ref = P.QuantileTransformer(n_quantiles=7, output_distribution='normal')
                expect(tag, 'rv', ref.fit_transform(X[::-1]), QTOL[dt], dt)
            for axis in (0, 1):
                for dist in ('uniform', 'normal'):
                    body.append(f'emit("fn{axis}{dist[0]}", quantile_transform(X, axis={axis}, n_quantiles=9, output_distribution="{dist}", subsample=None))')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        expect(tag, f'fn{axis}{dist[0]}', P.quantile_transform(X, axis=axis, n_quantiles=9, output_distribution=dist, subsample=None), QTOL[dt], dt)
        d.case(tag, '\n'.join(body))

# Seeded subsampling: integer seeds reseed per fit; a borrowed RandomState advances.
X = rng.randn(500, 2)
body = [f'var X = {d.input(X)}']
for seed in (0, 1, 123456, 4294967295):
    body += [f'var s{seed} = QuantileTransformer(n_quantiles=50, subsample=97, random_state={seed})',
             f's{seed}.fit(X)', f'emit("s{seed}", s{seed}.quantiles_)', f's{seed}.fit(X)', f'emit("s{seed}b", s{seed}.quantiles_)']
    ref = P.QuantileTransformer(n_quantiles=50, subsample=97, random_state=seed).fit(X)
    expect('seeded', f's{seed}', ref.quantiles_, EXACT, np.float64)
    expect('seeded', f's{seed}b', ref.fit(X).quantiles_, EXACT, np.float64)
body += ['var rs = RandomState(3)', 'var m = QuantileTransformer(n_quantiles=50, subsample=97)',
         'm.fit(X, rs)', 'emit("rs1", m.quantiles_)', 'm.fit(X, rs)', 'emit("rs2", m.quantiles_)',
         'emit_scalar("rs_next", rs.random_sample())']
state = np.random.RandomState(3)
ref = P.QuantileTransformer(n_quantiles=50, subsample=97, random_state=state)
expect('seeded', 'rs1', ref.fit(X).quantiles_, EXACT, np.float64)
expect('seeded', 'rs2', ref.fit(X).quantiles_, EXACT, np.float64)
expect('seeded', 'rs_next', state.random_sample(), EXACT, np.float64)
d.case('seeded', '\n'.join(body))

# Feature names.
d.case('qt_names', f'''var m = QuantileTransformer(n_quantiles=5)
var names: List[String] = ["a", "b", "c"]
m.fit_named({d.input(qdata["toy"])}, names)
emit_names("named", m.get_feature_names_out())
m.fit({d.input(qdata["toy"])})
emit_names("plain", m.get_feature_names_out())''')
expect('qt_names', 'named', ['a', 'b', 'c'], 'names')
expect('qt_names', 'plain', ['x0', 'x1', 'x2'], 'names')

# ---------------------------------------------------------------------------
# PowerTransformer


def boxcox_overflow_data(dt):
    """Data on which SciPy's boxcox_normmax constrains lambda (ymax guard)."""
    local = np.random.RandomState(0)
    for _ in range(50):
        x = (1e-3 * (1 + 0.01 * np.abs(local.randn(20)))).astype(dt)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            from scipy import stats
            stats.boxcox_normmax(x, method='mle')
        if any('optimal lambda' in str(w.message) for w in caught):
            return x.reshape(-1, 1)
    raise AssertionError('no overflow-guard dataset found')


pos = np.exp(rng.randn(50, 3) * [0.5, 1.0, 2.0])
pdata = {
    'lognormal': pos,
    'mixed': rng.randn(45, 3) * [1.0, 5.0, 0.1] + [0.5, -1.0, 0.0],
    'negative': -np.exp(rng.randn(30, 2)),
    'nan': np.where(rng.rand(40, 3) < 0.2, np.nan, np.exp(rng.randn(40, 3))),
    'constant_col': np.column_stack([np.full(10, 2.0), np.exp(rng.randn(10)), np.full(10, -2.0)]),
    'non_gaussian': (1e6 * np.array([0.6, 2.0, 3.0, 4.0] * 4 + [11, 12, 12, 16, 17, 20, 85, 90])).reshape(-1, 1),
    'skew': np.array([[1.0], [1.0], [1e10]]),
    'years': np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0, 2009.0, 1980.0, 1999.0, 2007.0, 1991.0]).reshape(-1, 1),
    'zeros_nan': np.array([[0.0, 1.0], [0.0, 2.0], [np.nan, 3.5], [0.0, 0.25]]),
    'two': np.array([[1.5, -3.0], [2.5, 4.0]]),
}
for dt in (np.float64, np.float32):
    sets = dict(pdata)
    sets['bc_guard'] = boxcox_overflow_data(dt)
    for key, raw in sets.items():
        X = raw.astype(dt)
        positive = np.nanmin(X) > 0
        methods = ['yeo-johnson'] + (['box-cox'] if positive else [])
        if key == 'bc_guard':
            methods = ['box-cox']
        tag = f'pt_{key}_{dt.__name__}'
        T = (X[: min(len(X), 6)] * 1.1).astype(dt)
        body = [f'var X = {d.input(X)}', f'var T = {d.input(T)}']
        for method in methods:
            for std in (True, False):
                n = f'{method[0]}{int(std)}'
                body += [f'var p{n} = PowerTransformer[{MOJO[dt]}]("{method}", standardize={std})',
                         f'emit("ft{n}", p{n}.fit_transform(X))', f'emit("l{n}", p{n}.lambdas_)',
                         f'var f{n} = PowerTransformer[{MOJO[dt]}]("{method}", standardize={std})',
                         f'f{n}.fit(X)', f'emit("fl{n}", f{n}.lambdas_)',
                         f'emit("t{n}", f{n}.transform(X))', f'emit("tt{n}", f{n}.transform(T))',
                         f'emit("i{n}", f{n}.inverse_transform(f{n}.transform(T)))']
                ref = P.PowerTransformer(method=method, standardize=std)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    expect(tag, f'ft{n}', ref.fit_transform(X), PTOL[dt], dt)
                    expect(tag, f'l{n}', ref.lambdas_, LTOL, dt)
                    ref = P.PowerTransformer(method=method, standardize=std).fit(X)
                    expect(tag, f'fl{n}', ref.lambdas_, LTOL, dt)
                    expect(tag, f't{n}', ref.transform(X), PTOL[dt], dt)
                    expect(tag, f'tt{n}', ref.transform(T), PTOL[dt], dt)
                    expect(tag, f'i{n}', ref.inverse_transform(ref.transform(T)), PTOL[dt], dt)
            body.append(f'emit("fn{method[0]}", power_transform(X, "{method}"))')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                expect(tag, f'fn{method[0]}', P.power_transform(X, method=method), PTOL[dt], dt)
        d.case(tag, '\n'.join(body))

# Transforms with fixed lambdas isolate transform arithmetic from the optimizer.
for dt in (np.float64, np.float32):
    X = np.column_stack([rng.randn(20) * 3, np.exp(rng.randn(20)), rng.randn(20)]).astype(dt)
    Xp = np.abs(X) + 0.1
    for method, data, lams in (('yeo-johnson', X, [0.0, 2.0, -0.7]), ('yeo-johnson', X, [1.0, 0.3, 2.5]),
                               ('box-cox', Xp, [0.0, 0.5, -1.3])):
        tag = f'fixed_{method[0]}_{"_".join(str(v) for v in lams).replace(".", "p").replace("-", "m")}_{dt.__name__}'
        lam = np.array(lams, dtype=dt)
        ref = P.PowerTransformer(method=method, standardize=False).fit(data)
        ref.lambdas_ = lam
        d.case(tag, f'''var X = {d.input(data)}
var m = PowerTransformer[{MOJO[dt]}]("{method}", standardize=False)
m.fit(X)
m.lambdas_ = {d.input(lam)}
var t = m.transform(X)
emit("t", t)
emit("i", m.inverse_transform(t))''')
        t = ref.transform(data)
        expect(tag, 't', t, QTOL[dt] if dt == np.float64 else dict(rtol=5e-7, atol=1e-7), dt)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expect(tag, 'i', ref.inverse_transform(t), QTOL[dt] if dt == np.float64 else dict(rtol=5e-7, atol=1e-7), dt)

# ---------------------------------------------------------------------------
# Errors

errors = {
    'qt_subsample': ('var m = QuantileTransformer(subsample=10)\nm.fit({ok})', 'number of quantiles cannot be greater'),
    'qt_nq': ('var m = QuantileTransformer(n_quantiles=0)\nm.fit({ok})', "'n_quantiles' parameter"),
    'qt_dist': ('var m = QuantileTransformer(output_distribution="beta")\nm.fit({ok})', "'output_distribution' parameter"),
    'qt_sub0': ('var m = QuantileTransformer(subsample=0, n_quantiles=1)\nm.fit({ok})', "'subsample' parameter"),
    'qt_seed': ('var m = QuantileTransformer(random_state=-1)\nm.fit({ok})', "'random_state' parameter"),
    'qt_inf': ('var m = QuantileTransformer()\nm.fit({inf})', 'infinity'),
    'qt_features': ('var m = QuantileTransformer(n_quantiles=3)\nm.fit({ok})\n_ = m.transform({wide})', 'X has 3 features, but QuantileTransformer is expecting 2'),
    'qt_inv_features': ('var m = QuantileTransformer(n_quantiles=3)\nm.fit({ok})\n_ = m.inverse_transform({wide})', 'X has 3 features'),
    'qt_unfitted': ('var m = QuantileTransformer()\n_ = m.transform({ok})', 'NotFittedError'),
    'qt_empty': ('var m = QuantileTransformer(n_quantiles=3)\nm.fit({empty})', '0 sample(s)'),
    'qt_axis': ('_ = quantile_transform({ok}, axis=2)', "'axis' parameter"),
    'pt_method': ('var m = PowerTransformer("log")\nm.fit({ok})', "'method' parameter"),
    'pt_bc_neg': ('var m = PowerTransformer("box-cox")\nm.fit({neg})', 'strictly positive'),
    'pt_bc_zero_transform': ('var m = PowerTransformer("box-cox")\nm.fit({ok})\n_ = m.transform({zero})', 'strictly positive'),
    'pt_bc_allnan': ('var m = PowerTransformer("box-cox")\nm.fit({allnan})', 'Column must not be all nan.'),
    'pt_yj_allnan': ('var m = PowerTransformer()\nm.fit({allnan})', 'not enough values to unpack'),
    'pt_bc_constant': ('var m = PowerTransformer("box-cox")\nm.fit({const})', 'Data must not be constant.'),
    'pt_features': ('var m = PowerTransformer()\nm.fit({ok})\n_ = m.transform({wide})', 'X has 3 features, but PowerTransformer is expecting 2'),
    'pt_inv_features': ('var m = PowerTransformer()\nm.fit({ok})\n_ = m.inverse_transform({wide})', 'X has 3 features'),
    'pt_unfitted': ('var m = PowerTransformer()\n_ = m.inverse_transform({ok})', 'NotFittedError'),
}
ok = np.abs(rng.randn(12, 2)) + 0.5
allnan = np.column_stack([np.full(4, np.nan), np.ones(4) + np.arange(4)])
inputs = dict(ok=d.input(ok), wide=d.input(np.ones((3, 3))), inf=d.input(np.array([[1.0, np.inf]])),
              empty=d.input(np.zeros((0, 2))), neg=d.input(-ok), zero=d.input(np.zeros((2, 2))),
              allnan=d.input(allnan), const=d.input(np.full((5, 2), 3.0)))
sk_errors = {
    'qt_subsample': lambda: P.QuantileTransformer(subsample=10).fit(ok),
    'qt_nq': lambda: P.QuantileTransformer(n_quantiles=0).fit(ok),
    'qt_dist': lambda: P.QuantileTransformer(output_distribution='beta').fit(ok),
    'qt_sub0': lambda: P.QuantileTransformer(subsample=0, n_quantiles=1).fit(ok),
    'qt_seed': lambda: P.QuantileTransformer(random_state=-1).fit(ok),
    'qt_inf': lambda: P.QuantileTransformer().fit(np.array([[1.0, np.inf]])),
    'qt_features': lambda: P.QuantileTransformer(n_quantiles=3).fit(ok).transform(np.ones((3, 3))),
    'qt_inv_features': lambda: P.QuantileTransformer(n_quantiles=3).fit(ok).inverse_transform(np.ones((3, 3))),
    'qt_unfitted': lambda: P.QuantileTransformer().transform(ok),
    'qt_empty': lambda: P.QuantileTransformer(n_quantiles=3).fit(np.zeros((0, 2))),
    'qt_axis': lambda: P.quantile_transform(ok, axis=2),
    'pt_method': lambda: P.PowerTransformer(method='log').fit(ok),
    'pt_bc_neg': lambda: P.PowerTransformer(method='box-cox').fit(-ok),
    'pt_bc_zero_transform': lambda: P.PowerTransformer(method='box-cox').fit(ok).transform(np.zeros((2, 2))),
    'pt_bc_allnan': lambda: P.PowerTransformer(method='box-cox').fit(allnan),
    'pt_yj_allnan': lambda: P.PowerTransformer().fit(allnan),
    'pt_bc_constant': lambda: P.PowerTransformer(method='box-cox').fit(np.full((5, 2), 3.0)),
    'pt_features': lambda: P.PowerTransformer().fit(ok).transform(np.ones((3, 3))),
    'pt_inv_features': lambda: P.PowerTransformer().fit(ok).inverse_transform(np.ones((3, 3))),
    'pt_unfitted': lambda: P.PowerTransformer().inverse_transform(ok),
}
for name, (code, _) in errors.items():
    d.case(f'err_{name}', code.format(**inputs))

out = d.run(imports='from moxie.preprocessing import QuantileTransformer, PowerTransformer, quantile_transform, power_transform')

for case, name, value, tol, dtype in checks:
    got = out.get(case, {})
    label = f'{case}.{name}'
    if 'error' in got:
        d.close(label, got['error'], 0)
        continue
    if tol == 'int':
        d.equal(label, got.get(name), int(value))
    elif tol == 'names':
        d.equal(label, got.get(name), value)
    else:
        expected = np.asarray(value)
        if DEBUG and not isinstance(got.get(name), str) and got.get(name) is not None:
            g = np.asarray(got[name], dtype=np.float64)
            e = expected.astype(np.float64)
            if g.shape == e.shape:
                f = np.isfinite(e) & np.isfinite(g)
                if f.any():
                    rel = np.max(np.abs(g[f] - e[f]) / np.maximum(np.abs(e[f]), 1e-300))
                    key = (name.rstrip('0123456789un'), dtype)
                    if rel > worst.get(key, (0, ''))[0]:
                        worst[key] = (rel, label)
        d.close(label, got.get(name, 'missing'), expected, dtype=dtype, **tol)
for name, (code, needle) in errors.items():
    try:
        sk_errors[name]()
        d._fail(f'err_{name}', 'sklearn did not raise')
    except Exception:  # the oracle raises as well
        pass
    d.raises(f'err_{name}', out[f'err_{name}'].get('error'), needle)
if DEBUG:
    for key, (rel, label) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        print(f'{rel:.3e} {label}')
d.finish()
