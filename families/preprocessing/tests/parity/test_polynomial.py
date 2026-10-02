"""Parity: PolynomialFeatures and SplineTransformer against pinned scikit-learn.

Tolerances (see docs/api/polynomial.md):
* PolynomialFeatures: exact in Float32 and Float64. Every output column is one
  product of a previous column and an input column, computed in the input
  dtype in sklearn's order, so results are bit-identical.
* SplineTransformer: Float64 exact for knots and basis values (the native code
  ports SciPy's de Boor recursion and NumPy's linspace/percentile/remainder
  arithmetic operation by operation). Float32 output is the same Float64 value
  rounded once, also exact. Derivative-based `linear` extrapolation uses the
  same recurrences and is compared exactly as well.
"""
import math

import numpy as np

from harness import Driver, oracle

sk = oracle()
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

rng = np.random.RandomState(0)
d = Driver('polynomial')
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}
expected = {}   # (case, name) -> array | list | int | ('raises', needle)


def mojo_bool(v):
    return 'True' if v else 'False'


def record(tag, name, value):
    expected[(tag, name)] = value


def sk_error(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - message fragment recorded
        return e
    return None


# ---------------------------------------------------------------------------
# PolynomialFeatures

poly_data = {
    'gauss3': rng.randn(6, 3),
    'one': rng.randn(5, 1) * 3,
    'wide': rng.randn(3, 5),
    'ints': np.arange(12.0).reshape(4, 3) - 5,
}
poly_configs = []
for degree in (0, 1, 2, 3, 4, (2, 3), (0, 2), (1, 1), (3, 3), (2, 2), (1, 3), (0, 0)):
    for interaction_only in (False, True):
        for include_bias in (True, False):
            poly_configs.append((degree, interaction_only, include_bias))

k = 0
for dt in (np.float64, np.float32):
    for key, raw in poly_data.items():
        X = raw.astype(dt)
        T = d.input(X)
        tag = f'poly_{key}_{dt.__name__}'
        body = [f'var X = {T}']
        for degree, interaction_only, include_bias in poly_configs:
            order = 'F' if k % 3 == 0 else 'C'
            k += 1
            name = f'{degree}_{int(interaction_only)}{int(include_bias)}'.replace(' ', '').replace('(', '').replace(')', '').replace(',', 'x')
            degree_text = f'({degree[0]}, {degree[1]})' if isinstance(degree, tuple) else str(degree)
            est = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
            err = sk_error(lambda: est.fit(X))
            body.append(f'var m_{name} = PolynomialFeatures[{MOJO[dt]}]({degree_text}, interaction_only={mojo_bool(interaction_only)}, include_bias={mojo_bool(include_bias)}, order="{order}")')
            if err is not None:
                body.append(f'try:\n    m_{name}.fit(X)\n    print("I e_{name} 0")\nexcept e:\n    emit_error("e_{name}", e)')
                record(tag, f'e_{name}', ('raises', str(err).split('\n')[0][:40]))
                continue
            out = est.transform(X)
            body.append(f'var o_{name} = m_{name}.fit_transform(X)')
            body.append(f'emit("o_{name}", o_{name})')
            if out.shape[0] > 1 and out.shape[1] > 1:
                body.append(f'emit_int("f_{name}", 1 if o_{name}.flags.F_CONTIGUOUS and not o_{name}.flags.C_CONTIGUOUS else 0)')
                record(tag, f'f_{name}', int(out.flags.f_contiguous and not out.flags.c_contiguous))
            body.append(f'emit_int("n_{name}", m_{name}.n_output_features_)')
            body.append(f'emit("p_{name}", m_{name}.powers_())')
            body.append(f'emit_names("names_{name}", m_{name}.get_feature_names_out())')
            record(tag, f'o_{name}', out)
            record(tag, f'n_{name}', est.n_output_features_)
            if est.n_output_features_ == 0:
                # sklearn's powers_/get_feature_names_out raise on an empty
                # expansion (np.vstack of nothing); natively they are empty.
                assert sk_error(lambda: est.powers_) is not None
                record(tag, f'p_{name}', np.zeros((0, X.shape[1]), dtype=np.int64))
                record(tag, f'names_{name}', [])
            else:
                record(tag, f'p_{name}', est.powers_)
                record(tag, f'names_{name}', list(est.get_feature_names_out()))
        d.case(tag, '\n'.join(body))

# Named features, explicit input_features, repeat fit and strided input.
base = rng.randn(12, 6)
view = base[::2, 5::-2]  # strides (12, -2) elements, offset 5
names = ['a', 'b', 'c']
# fit_named records feature_names_in_; sklearn's equivalent output is
# get_feature_names_out(names) on an array fit.
record('poly_named', 'names', list(PolynomialFeatures(degree=3).fit(view).get_feature_names_out(names)))
record('poly_named', 'explicit', list(PolynomialFeatures(degree=2, interaction_only=True).fit(view).get_feature_names_out(['u', 'v', 'w'])))
record('poly_named', 'strided', PolynomialFeatures(degree=3).fit_transform(view))
record('poly_named', 'refit', PolynomialFeatures(degree=3).fit(view).fit(view[:, :2]).transform(view[:2, :2]))
d.case('poly_named', f'''var B = {d.input(base)}
var V = B.view_with_layout(Shape(6, 3), NDArrayStrides(12, -2), 5)
var m = PolynomialFeatures(3)
var names: List[String] = ["a", "b", "c"]
m.fit_named(V, names)
emit_names("names", m.get_feature_names_out())
var m2 = PolynomialFeatures(2, interaction_only=True)
m2.fit(V)
var given: List[String] = ["u", "v", "w"]
emit_names("explicit", m2.get_feature_names_out(given^))
emit("strided", m.transform(V))
m.fit(V)
var V2 = B.view_with_layout(Shape(6, 2), NDArrayStrides(12, -2), 5)
m.fit(V2)
var V3 = B.view_with_layout(Shape(2, 2), NDArrayStrides(12, -2), 5)
emit("refit", m.transform(V3))''')

# Error cases.
X3 = rng.randn(4, 3)
huge = np.ones((1, 100))
poly_errors = {
    'neg': ('-1', lambda: PolynomialFeatures(degree=-1).fit(X3), 'X'),
    'tuple_neg': ('(-1, 2)', lambda: PolynomialFeatures(degree=(-1, 2)).fit(X3), 'X'),
    'tuple_order': ('(3, 2)', lambda: PolynomialFeatures(degree=(3, 2)).fit(X3), 'X'),
    'zero_nobias': ('0, include_bias=False', lambda: PolynomialFeatures(degree=0, include_bias=False).fit(X3), 'X'),
    'zero_tuple_nobias': ('(0, 0), include_bias=False', lambda: PolynomialFeatures(degree=(0, 0), include_bias=False).fit(X3), 'X'),
    'order': ('2, order="X"', lambda: PolynomialFeatures(order='X').fit(X3), 'X'),
    'overflow': ('(0, 100)', lambda: PolynomialFeatures(degree=(0, 100)).fit(huge), 'H'),
}
lines = [f'var X = {d.input(X3)}', f'var H = {d.input(huge)}', f'var N = {d.input(np.array([[1.0, np.nan, 2.0]]))}']
for name, (args, fn, data) in poly_errors.items():
    err = sk_error(fn)
    assert err is not None, name
    fragment = str(err)[:60] if name != 'overflow' else str(err).split(' features ')[0]
    record('poly_errors', name, ('raises', fragment))
    lines.append(f'try:\n    var m = PolynomialFeatures({args})\n    m.fit({data})\n    print("I {name} 0")\nexcept e:\n    emit_error("{name}", e)')
fitted = PolynomialFeatures().fit(X3)
for name, fn, body in [
    ('nan', lambda: fitted.transform(np.array([[1.0, np.nan, 2.0]])), 'var m = PolynomialFeatures()\nm.fit(X)\n_ = m.transform(N)'),
    ('nan_fit', lambda: PolynomialFeatures().fit(np.array([[1.0, np.nan, 2.0]])), 'var m = PolynomialFeatures()\nm.fit(N)'),
    ('features', lambda: fitted.transform(X3[:, :2]), f'var m = PolynomialFeatures()\nm.fit(X)\n_ = m.transform({d.input(X3[:, :2])})'),
    ('unfitted', None, 'var m = PolynomialFeatures()\n_ = m.transform(X)'),
    ('unfitted_names', None, 'var m = PolynomialFeatures()\n_ = m.get_feature_names_out()'),
]:
    err = sk_error(fn) if fn else None
    needle = str(err).split('\n')[0][:30] if err is not None else 'NotFittedError'
    record('poly_errors', name, ('raises', needle))
    indented = '\n'.join('    ' + s for s in body.split('\n'))
    lines.append(f'try:\n{indented}\n    print("I {name} 0")\nexcept e:\n    emit_error("{name}", e)')
d.case('poly_errors', '\n'.join(lines))

# ---------------------------------------------------------------------------
# SplineTransformer

def spline_case(tag, X, Xt, kwargs, *, weights=None, knots_array=None, scalar_weight=None):
    dt = X.dtype.type
    kw = dict(kwargs)
    est = SplineTransformer(**kw, **({'knots': knots_array} if knots_array is not None else {}))
    fit_kw = {}
    if weights is not None:
        fit_kw['sample_weight'] = weights
    if scalar_weight is not None:
        fit_kw['sample_weight'] = scalar_weight
    lines = [f'var X = {d.input(X)}', f'var Xt = {d.input(Xt)}']
    args = []
    for key, value in kw.items():
        if isinstance(value, bool):
            args.append(f'{key}={mojo_bool(value)}')
        elif isinstance(value, str):
            args.append(f'{key}="{value}"')
        else:
            args.append(f'{key}={value}')
    if knots_array is not None:
        lines.append(f'var K = {d.input(np.asarray(knots_array, dtype=np.float64))}')
        args.append('knots=K')
    lines.append(f'var m = SplineTransformer[{MOJO[dt]}]({", ".join(args)})')
    if weights is not None:
        lines.append(f'var W = {d.input(weights)}')
        fit_call = 'm.fit(X, sample_weight=W)'
    elif scalar_weight is not None:
        fit_call = f'm.fit(X, sample_weight=Float64({scalar_weight}))'
    else:
        fit_call = 'm.fit(X)'
    err = sk_error(lambda: est.fit(X, **fit_kw))
    if err is not None:
        lines.append(f'try:\n    {fit_call}\n    print("I fit 0")\nexcept e:\n    emit_error("fit", e)')
        record(tag, 'fit', ('raises', str(err).split('\n')[0][:50]))
        d.case(tag, '\n'.join(lines))
        return
    lines.append(fit_call)
    lines.append('emit_int("nout", m.n_features_out_)')
    lines.append('emit_names("names", m.get_feature_names_out())')
    record(tag, 'nout', est.n_features_out_)
    record(tag, 'names', list(est.get_feature_names_out()))
    for j, spl in enumerate(est.bsplines_):
        lines.append(f'emit("t{j}", m.bsplines_[{j}].t)')
        record(tag, f't{j}', np.asarray(spl.t, dtype=np.float64))
    lines.append('emit("c", m.bsplines_[0].coefficients())')
    record(tag, 'c', est.bsplines_[0].c)
    if X.dtype == np.float32 and kw.get('knots') == 'quantile' and fit_kw:
        # sklearn keeps Float32 knots here and SciPy then rejects them at
        # transform; natively the same knots are evaluated in Float64.
        assert 'dtype' in str(sk_error(lambda: est.transform(X)))
        from scipy.interpolate import BSpline
        est.bsplines_ = [BSpline.construct_fast(s.t.astype(np.float64), s.c, s.k, extrapolate=s.extrapolate) for s in est.bsplines_]
    for name, data in (('train', X), ('test', Xt)):
        err = sk_error(lambda: est.transform(data))
        var = 'X' if name == 'train' else 'Xt'
        if kw.get('extrapolation') == 'linear' and kw.get('degree', 3) <= 1 and X.shape[1] > 1:
            # sklearn defect (not reproduced): its linear branch increments the
            # loop-shared `degree` for degree <= 1, so features after the first
            # use shifted boundary knots. Reference: each feature fit alone.
            parts = [SplineTransformer(**kw).fit(X[:, [j]], **fit_kw).transform(data[:, [j]]) for j in range(X.shape[1])]
            err, out = None, np.hstack(parts)
        elif kw.get('extrapolation') == 'constant' and kw.get('degree', 3) == 0 and err is not None and 'broadcast' in str(err):
            # sklearn defect (not reproduced): degree 0 above the range assigns
            # f_max[-0:] (all splines) into zero columns and fails. Natively the
            # rows are zero, exactly as sklearn encodes rows below the range.
            low = np.array([spl.t[0] for spl in est.bsplines_])
            high = np.array([spl.t[-1] for spl in est.bsplines_])
            err, out = None, est.transform(np.where(data > high, low - 1, data).astype(data.dtype))
        elif err is None:
            out = est.transform(data)
        if err is not None:
            lines.append(f'try:\n    _ = m.transform({var})\n    print("I {name} 0")\nexcept e:\n    emit_error("{name}", e)')
            record(tag, name, ('raises', str(err).split('\n')[0][:50]))
        else:
            lines.append(f'emit("{name}", m.transform({var}))')
            record(tag, name, out)
    d.case(tag, '\n'.join(lines))


Xs = {
    'gauss': rng.randn(20, 3),
    'ties': rng.randint(0, 4, size=(15, 2)).astype(np.float64),
    'skew': np.exp(rng.randn(25, 2)),
}
count = 0
for dt in (np.float64, np.float32):
    for key, raw in Xs.items():
        X = raw.astype(dt)
        lo, hi = raw.min(axis=0), raw.max(axis=0)
        span = hi - lo
        Xt = np.vstack([lo - 0.7 * span, lo - 1e-3, lo, (lo + hi) / 2, hi, hi + 0.4 * span, hi + 3 * span, raw[:3]]).astype(dt)
        for extrapolation in ('constant', 'linear', 'continue', 'periodic', 'error'):
            for degree, n_knots in ((0, 3), (1, 2), (2, 4), (3, 5), (4, 6), (3, 3)):
                for knots in ('uniform', 'quantile'):
                    count += 1
                    include_bias = count % 2 == 0
                    order = 'F' if count % 5 == 0 else 'C'
                    kwargs = dict(n_knots=n_knots, degree=degree, knots=knots, extrapolation=extrapolation, include_bias=include_bias, order=order)
                    spline_case(f'sp_{key}_{dt.__name__}_{extrapolation}_{degree}_{n_knots}_{knots}', X, Xt, kwargs)

# Weighted knots (uniform ignores zero weights; quantile uses _weighted_percentile).
for dt in (np.float64, np.float32):
    X = rng.randn(18, 2).astype(dt)
    Xt = np.vstack([X.min(axis=0) - 1, X, X.max(axis=0) + 1]).astype(dt)
    w = rng.rand(18).astype(dt)
    w[[0, 5, 6]] = 0
    for knots in ('uniform', 'quantile'):
        for extrapolation in ('constant', 'linear', 'periodic'):
            spline_case(f'sw_{dt.__name__}_{knots}_{extrapolation}', X, Xt, dict(n_knots=4, degree=3, knots=knots, extrapolation=extrapolation), weights=w)
    spline_case(f'sw_scalar_{dt.__name__}', X, Xt, dict(n_knots=4, knots='quantile'), scalar_weight=2.5)
    wi = rng.randint(0, 3, 18).astype(dt)
    wi[0] = 1
    spline_case(f'sw_int_{dt.__name__}', X, Xt, dict(n_knots=5, knots='quantile', extrapolation='continue'), weights=wi)

# Negative weights (allowed by sklearn) make the cumulative weights non-monotone;
# NumPy's binary search is reproduced step by step.
Xw = rng.randn(12, 2)
wn = rng.randn(12)
wn[3] = 4.0
for knots in ('uniform', 'quantile'):
    spline_case(f'sw_negative_{knots}', Xw, Xw, dict(n_knots=4, knots=knots), weights=wn)

# sklearn's docstring example.
spline_case('sd_docstring', np.arange(6.0).reshape(6, 1), np.arange(6.0).reshape(6, 1), dict(degree=2, n_knots=3))

# sklearn's documented base-knot example (test_spline_transformer_get_base_knot_positions).
Xb = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]], dtype=np.float64)
wb = np.array([0, 0, 1, 1, 0, 3, 1], dtype=np.float64)
for knots in ('uniform', 'quantile'):
    spline_case(f'sb_{knots}', Xb, Xb, dict(n_knots=3, knots=knots))
    spline_case(f'sb_{knots}_w', Xb, Xb, dict(n_knots=3, knots=knots), weights=wb)
spline_case('sb_uniform_4', Xb, Xb, dict(n_knots=4))

# Explicit knots, including integer-valued and periodic ones.
Xk = np.arange(20, dtype=np.float64).reshape(10, 2)
Kk = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
for extrapolation in ('continue', 'periodic', 'constant', 'linear', 'error'):
    spline_case(f'sk_{extrapolation}', Xk, np.vstack([Xk, [[-3, 30]]]), dict(degree=3, extrapolation=extrapolation), knots_array=Kk)
    spline_case(f'sk32_{extrapolation}', Xk.astype(np.float32), Xk.astype(np.float32) * 1.5 - 2, dict(degree=2, extrapolation=extrapolation, include_bias=False), knots_array=Kk)

# Constant feature (all knots equal) and periodic with a single repeated value.
Xc = np.column_stack([np.full(6, 2.5), np.linspace(-1, 1, 6)])
for extrapolation in ('constant', 'linear', 'continue', 'periodic', 'error'):
    spline_case(f'sc_{extrapolation}', Xc, np.array([[2.5, 0.0], [3.0, 0.5], [1.0, -2.0]]), dict(degree=2, n_knots=3, extrapolation=extrapolation))

# handle_missing='zeros': NaN in fit and transform, an all-NaN column, infinity.
Xn = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [4, 4]], dtype=np.float64)
Xn[3, 0] = np.nan
Xall = np.array([[np.nan, np.nan], [np.nan, 1]])
Xinf = np.array([[np.inf, 2.0], [-np.inf, np.nan], [2.0, 3.0]])
for extrapolation in ('error', 'constant', 'linear', 'continue', 'periodic'):
    for knots in ('uniform', 'quantile'):
        for dt in (np.float64, np.float32):
            spline_case(f'sn_{extrapolation}_{knots}_{dt.__name__}', Xn.astype(dt), np.vstack([Xn, [[np.nan, np.nan]]]).astype(dt), dict(degree=2, n_knots=3, knots=knots, extrapolation=extrapolation, handle_missing='zeros'))
    spline_case(f'sn_all_{extrapolation}', Xall, Xall, dict(degree=2, n_knots=3, extrapolation=extrapolation, handle_missing='zeros'))
    spline_case(f'sn_inf_{extrapolation}', Xn[[0, 1, 2, 4]], Xinf, dict(degree=2, n_knots=3, extrapolation=extrapolation, handle_missing='zeros'))
    spline_case(f'sn_w_{extrapolation}', Xn, Xn, dict(degree=2, n_knots=3, knots='quantile', extrapolation=extrapolation, handle_missing='zeros'), weights=np.array([1.0, 2.0, 0.5, 3.0, 1.0]))

# Error cases.
X2 = np.array([[1.0], [2.0]])
spline_case('se_knots_rows', X2, X2, {}, knots_array=[[1.0]])
spline_case('se_knots_cols', X2, X2, {}, knots_array=[[1.0, 1.0], [2.0, 2.0]])
spline_case('se_knots_sorted', X2, X2, {}, knots_array=[[1.0], [0.0]])
spline_case('se_knots_nan', X2, X2, {}, knots_array=[[1.0], [np.nan]])
spline_case('se_periodic', X2, X2, dict(n_knots=3, degree=3, extrapolation='periodic'))
spline_case('se_n_knots', X2, X2, dict(n_knots=1))
spline_case('se_degree', X2, X2, dict(degree=-1))
spline_case('se_knots_str', X2, X2, dict(knots='middle'))
spline_case('se_extrapolation', X2, X2, dict(extrapolation='mirror'))
spline_case('se_order', X2, X2, dict(order='K'))
spline_case('se_missing', X2, X2, dict(handle_missing='mean'))
spline_case('se_one_sample', X2[:1], X2, {})
spline_case('se_nan_error', np.array([[1.0], [np.nan], [2.0]]), X2, {})
spline_case('se_inf', np.array([[1.0], [np.inf], [2.0]]), X2, {})
spline_case('se_nan_transform', np.array([[1.0], [3.0], [2.0]]), np.array([[np.nan]]), {})
spline_case('se_features', np.array([[1.0], [3.0], [2.0]]), np.ones((2, 2)), {})
for name, w in [('w_nan', np.array([1.0, np.nan])), ('w_inf', np.array([1.0, np.inf])), ('w_zero', np.zeros(2)), ('w_len', np.ones(3))]:
    spline_case(f'se_{name}', X2, X2, {}, weights=w)
spline_case('se_w_scalar_zero', X2, X2, {}, scalar_weight=0.0)
lines = [f'var X = {d.input(X2)}']
lines.append('try:\n    var m = SplineTransformer(sparse_output=True)\n    m.fit(X)\n    print("I sparse 0")\nexcept e:\n    emit_error("sparse", e)')
lines.append('try:\n    var m = SplineTransformer()\n    _ = m.transform(X)\n    print("I unfitted 0")\nexcept e:\n    emit_error("unfitted", e)')
d.case('se_native', '\n'.join(lines))
record('se_native', 'sparse', ('raises', 'sparse output is deferred'))
record('se_native', 'unfitted', ('raises', 'NotFittedError'))

# BSpline.evaluate (value and first derivative) against SciPy, and exact fmod.
est = SplineTransformer(n_knots=5, degree=3, extrapolation='continue').fit(Xs['gauss'])
spl = est.bsplines_[1]
points = np.linspace(spl.t[0] - 1, spl.t[-1] + 1, 23)
lines = [f'var X = {d.input(Xs["gauss"])}', 'var m = SplineTransformer(n_knots=5, degree=3, extrapolation="continue")', 'm.fit(X)']
for q, x in enumerate(points):
    for nu in (0, 1, 2):
        lines.append(f'emit("e{q}_{nu}", m.bsplines_[1].evaluate({float(x)!r}, {nu}))')
        record('bspline_eval', f'e{q}_{nu}', spl(x, nu=nu))
a = np.concatenate([rng.randn(40) * 10 ** rng.uniform(-3, 8, 40), [1e300, -1e300, 5e-324, 0.1, -0.0, 7.5, -7.5, 2.0 ** 60 + 1]])
b = np.concatenate([rng.randn(40) * 10 ** rng.uniform(-6, 3, 40), [3.0, 7.0, 1e-310, 0.01, 2.0, 2.0, -2.0, 3.0]])
lines.append(f'var A = {d.input(a)}')
lines.append(f'var Bv = {d.input(b)}')
lines.append('var fm = zeros[DType.float64](Shape(A.shape[0]))\nvar rm = zeros[DType.float64](Shape(A.shape[0]))')
lines.append('for i in range(A.shape[0]):\n    fm.unsafe_set(i, _fmod(A.unsafe_get(i), Bv.unsafe_get(i)))\n    rm.unsafe_set(i, _remainder(A.unsafe_get(i), Bv.unsafe_get(i)))')
lines.append('emit("fmod", fm)\nemit("remainder", rm)')
record('bspline_eval', 'fmod', np.fmod(a, b))
record('bspline_eval', 'remainder', np.remainder(a, b))
d.case('bspline_eval', '\n'.join(lines))

out = d.run(imports='from numojo import NDArrayStrides\n'
                    'from moxie.preprocessing import PolynomialFeatures, SplineTransformer\n'
                    'from moxie.preprocessing._polynomial import _fmod, _remainder')

for (tag, name), value in expected.items():
    got = out.get(tag, {})
    label = f'{tag}.{name}'
    if 'error' in got:
        d.close(label, got['error'], np.zeros(1))
        continue
    if isinstance(value, tuple) and value and value[0] == 'raises':
        d.raises(label, got.get(name), value[1])
    elif isinstance(value, list):
        d.equal(label, got.get(name), value)
    elif isinstance(value, int):
        d.equal(label, got.get(name), value)
    else:
        value = np.asarray(value)
        if name in ('fmod', 'remainder'):
            # Signed zeros matter for remainder; compare bit patterns too.
            g = got.get(name, 'missing')
            d.close(label, g, value, rtol=0, atol=0)
            if not isinstance(g, str):
                d.equal(label + '.signbit', list(np.signbit(g)), list(np.signbit(value)))
            continue
        d.close(label, got.get(name, 'missing'), value, rtol=0, atol=0, dtype=value.dtype)
d.finish()
