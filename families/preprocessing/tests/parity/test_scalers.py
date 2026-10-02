"""Parity: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler and functions.

Tolerances. Learned statistics: Float64 rtol 1e-11 (weighted sums use serial
accumulation where sklearn calls BLAS matmul); extrema and quantiles are exact.
Transforms: Float64 rtol 1e-11 / atol 1e-12, Float32 rtol 2e-6 / atol 2e-6,
because parameters are rounded to Float32 exactly as sklearn's astype but may
differ from sklearn's in the last Float64 bit before rounding.
Weighted transforms add, per column j, the recursive-summation forward bound
n * eps64 * sum_i |w_i x_ij| / |sum_i w_i| / scale_j: sklearn's weighted sum is
a BLAS matmul whose order and FMA use are implementation-defined, so only the
rounding bound (amplified by |mean| / std on ill-conditioned columns) is valid.
"""
import numpy as np

from harness import Driver, oracle

oracle()
from sklearn import preprocessing as P

rng = np.random.RandomState(42)
d = Driver('scalers')
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}
STAT = dict(rtol=1e-11, atol=1e-13)
TOL = {np.float64: dict(rtol=1e-11, atol=1e-12), np.float32: dict(rtol=2e-6, atol=2e-6)}
checks = []  # (case, name, expected, kind)


def expect(case, name, value, kind='out', dtype=None):
    checks.append((case, name, value, kind, dtype))


d.prelude('''
from moxie.preprocessing._scalers import SampleCount

def emit_count(name: String, c: SampleCount) raises:
    emit(name, c.values)
    emit_int(name + "_scalar", Int(c.is_scalar))
    emit_int(name + "_integer", Int(c.is_integer))

def emit_optional(name: String, value: Optional[NDArray[DType.float64]]) raises:
    if value:
        emit(name, value.value())
    else:
        emit_int(name + "_none", 1)

def emit_optional32(name: String, value: Optional[NDArray[DType.float32]]) raises:
    if value:
        emit(name, value.value())
    else:
        emit_int(name + "_none", 1)

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
''')


def emit_count_expected(value, features):
    # SampleCount always carries one count per feature; a scalar is broadcast.
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (features,))
    return arr, int(np.ndim(value) == 0), int(np.asarray(value).dtype.kind == 'i')


def weighted_atol(X, w, scale):
    # Forward error bound of the weighted mean, in transformed units.
    X = X.astype(np.float64)
    w = np.asarray(w, dtype=np.float64)
    terms = np.nansum(np.abs(w[:, None] * X), axis=0) / abs(w.sum())
    return X.shape[0] * np.finfo(np.float64).eps * terms / scale


datasets = {
    'gauss': rng.randn(9, 3) * [1, 10, 1e-3] + [0, -5, 1e4],
    'nan': np.where(rng.rand(10, 4) < 0.25, np.nan, rng.randn(10, 4)),
    'allnan_col': np.column_stack([rng.randn(6), np.full(6, np.nan), np.ones(6)]),
    'constant': np.column_stack([np.full(7, 3.3), np.full(7, 1e8) + np.arange(7) * 1e-8, rng.randn(7)]),
    'single': np.array([[2.5, -1.0]]),
    'overflow': np.array([[1e308], [0], [1e308], [-1e308], [-1e308], [0], [0], [0]]),
    'tall': rng.rand(300, 2) * 1e3,
}
for dt in (np.float64, np.float32):
    for key, raw in datasets.items():
        if key == 'overflow' and dt == np.float32:
            continue
        X = raw.astype(dt)
        tag = f'std_{key}_{dt.__name__}'
        body = [f'var X = {d.input(X)}']
        for wm in (True, False):
            for ws in (True, False):
                n = f'{int(wm)}{int(ws)}'
                body += [f'var s{n} = StandardScaler[{MOJO[dt]}](with_mean={wm}, with_std={ws})',
                         f'emit("t{n}", s{n}.fit_transform(X))',
                         f'emit("i{n}", s{n}.inverse_transform(s{n}.transform(X)))',
                         f'emit_count("c{n}", s{n}.n_samples_seen_)',
                         f'emit_optional("m{n}", s{n}.mean_)', f'emit_optional("v{n}", s{n}.var_)',
                         f'emit_optional("sc{n}", s{n}.scale_)']
                ref = P.StandardScaler(with_mean=wm, with_std=ws).fit(X)
                expect(tag, f't{n}', ref.transform(X), dtype=dt)
                expect(tag, f'i{n}', ref.inverse_transform(ref.transform(X)), dtype=dt)
                expect(tag, f'c{n}', ref.n_samples_seen_, 'count', dtype=X.shape[1])
                for attr, short in (('mean_', 'm'), ('var_', 'v'), ('scale_', 'sc')):
                    expect(tag, f'{short}{n}', getattr(ref, attr), 'optional')
        if X.shape[0] > 1:
            # partial_fit in two batches, unweighted then weighted, and scalar weights
            h = X.shape[0] // 2
            w = (rng.rand(X.shape[0]) * 3).astype(dt)
            w[0] = -0.5  # signed weights are accepted
            body += [f'var A = {d.input(X[:h])}', f'var B = {d.input(X[h:])}', f'var W = {d.input(w[h:])}',
                     f'var Wfull = {d.input(w)}',
                     f'var pf = StandardScaler[{MOJO[dt]}]()', 'pf.partial_fit(A)', 'pf.partial_fit(B, sample_weight=W)',
                     'emit("pf_t", pf.transform(X))', 'emit_count("pf_c", pf.n_samples_seen_)',
                     'emit_optional("pf_v", pf.var_)',
                     f'var wf = StandardScaler[{MOJO[dt]}]()', 'wf.fit(X, sample_weight=Wfull)',
                     'emit("wf_t", wf.transform(X))', 'emit_count("wf_c", wf.n_samples_seen_)',
                     'emit_optional("wf_m", wf.mean_)', 'emit_optional("wf_v", wf.var_)',
                     f'var sw = StandardScaler[{MOJO[dt]}]()', 'sw.fit(X, sample_weight=2.5)',
                     'emit_count("sw_c", sw.n_samples_seen_)', 'emit_optional("sw_v", sw.var_)',
                     f'var fo = StandardScaler[{MOJO[dt]}]()', 'fo.fit(fortran(X))',
                     'emit("fo_t", fo.transform(fortran(X)))', 'emit_optional("fo_m", fo.mean_)',
                     f'var rv = StandardScaler[{MOJO[dt]}]()', 'rv.fit(reversed_rows(X))',
                     'emit("rv_t", rv.transform(reversed_rows(X)))']
            pf = P.StandardScaler().partial_fit(X[:h]).partial_fit(X[h:], sample_weight=w[h:])
            wpf = np.concatenate([np.ones(h, dtype=dt), w[h:]])
            expect(tag, 'pf_t', pf.transform(X), 'weighted', dtype=(dt, weighted_atol(X, wpf, pf.scale_)))
            expect(tag, 'pf_c', pf.n_samples_seen_, 'count', dtype=X.shape[1])
            expect(tag, 'pf_v', pf.var_, 'optional')
            wf = P.StandardScaler().fit(X, sample_weight=w)
            expect(tag, 'wf_t', wf.transform(X), 'weighted', dtype=(dt, weighted_atol(X, w, wf.scale_)))
            expect(tag, 'wf_c', wf.n_samples_seen_, 'count', dtype=X.shape[1])
            expect(tag, 'wf_m', wf.mean_, 'optional')
            expect(tag, 'wf_v', wf.var_, 'optional')
            sw = P.StandardScaler().fit(X, sample_weight=2.5)
            expect(tag, 'sw_c', sw.n_samples_seen_, 'count', dtype=X.shape[1])
            expect(tag, 'sw_v', sw.var_, 'optional')
            Xf = np.asfortranarray(X)
            fo = P.StandardScaler().fit(Xf)
            expect(tag, 'fo_t', fo.transform(Xf), dtype=dt)
            expect(tag, 'fo_m', fo.mean_, 'optional')
            Xr = X[::-1]
            rv = P.StandardScaler().fit(Xr)
            expect(tag, 'rv_t', rv.transform(Xr), dtype=dt)
        d.case(tag, '\n'.join(body))

        if key == 'overflow':
            continue
        # MinMax / MaxAbs / Robust
        tag = f'other_{key}_{dt.__name__}'
        body = [f'var X = {d.input(X)}']
        for k, (fr, clip) in enumerate([((0, 1), False), ((-3.0, 2.5), True), ((1e-3, 1e-3 + 1e-6), False)]):
            body += [f'var mm{k} = MinMaxScaler[{MOJO[dt]}](({float(fr[0])}, {float(fr[1])}), clip={clip})',
                     f'emit("mm{k}", mm{k}.fit_transform(X))',
                     f'emit("mmi{k}", mm{k}.inverse_transform(mm{k}.transform(X)))',
                     f'emit("mms{k}", mm{k}.scale_)', f'emit("mmn{k}", mm{k}.min_)',
                     f'emit("mmr{k}", mm{k}.data_range_)', f'emit_int("mmc{k}", mm{k}.n_samples_seen_)']
            ref = P.MinMaxScaler(feature_range=fr, clip=clip).fit(X)
            expect(tag, f'mm{k}', ref.transform(X), dtype=dt)
            expect(tag, f'mmi{k}', ref.inverse_transform(ref.transform(X)), dtype=dt)
            expect(tag, f'mms{k}', ref.scale_, dtype=dt)
            expect(tag, f'mmn{k}', ref.min_, dtype=dt)
            expect(tag, f'mmr{k}', ref.data_range_, 'exact', dtype=dt)
            expect(tag, f'mmc{k}', ref.n_samples_seen_, 'int')
        for clip in (False, True):
            body += [f'var ma{int(clip)} = MaxAbsScaler[{MOJO[dt]}](clip={clip})',
                     f'ma{int(clip)}.fit(X)',
                     f'emit("ma{int(clip)}", ma{int(clip)}.transform(X * 2))' if False else f'emit("ma{int(clip)}", ma{int(clip)}.transform(X))',
                     f'emit("mai{int(clip)}", ma{int(clip)}.inverse_transform(X))',
                     f'emit("mam{int(clip)}", ma{int(clip)}.max_abs_)']
            ref = P.MaxAbsScaler(clip=clip).fit(X)
            expect(tag, f'ma{int(clip)}', ref.transform(X), dtype=dt)
            expect(tag, f'mai{int(clip)}', ref.inverse_transform(X), dtype=dt)
            expect(tag, f'mam{int(clip)}', ref.max_abs_, 'exact', dtype=dt)
        if X.shape[0] > 1:
            h = X.shape[0] // 2
            body += [f'var pmm = MinMaxScaler[{MOJO[dt]}]()', f'pmm.partial_fit({d.input(X[:h])})',
                     f'pmm.partial_fit({d.input(X[h:])})', 'emit("pmm", pmm.transform(X))',
                     'emit_int("pmmc", pmm.n_samples_seen_)',
                     f'var pma = MaxAbsScaler[{MOJO[dt]}]()', f'pma.partial_fit({d.input(X[:h])})',
                     f'pma.partial_fit({d.input(X[h:])})', 'emit("pma", pma.transform(X))']
            ref = P.MinMaxScaler().partial_fit(X[:h]).partial_fit(X[h:])
            expect(tag, 'pmm', ref.transform(X), dtype=dt)
            expect(tag, 'pmmc', ref.n_samples_seen_, 'int')
            ref = P.MaxAbsScaler().partial_fit(X[:h]).partial_fit(X[h:])
            expect(tag, 'pma', ref.transform(X), dtype=dt)
        for k, (wc, wsc, qr, uv) in enumerate([(True, True, (25.0, 75.0), False), (False, True, (10.0, 90.0), True),
                                               (True, False, (25.0, 75.0), False), (True, True, (0.0, 100.0), True),
                                               (True, True, (40.0, 40.0), False)]):
            body += [f'var rb{k} = RobustScaler[{MOJO[dt]}](with_centering={wc}, with_scaling={wsc}, quantile_range=({qr[0]}, {qr[1]}), unit_variance={uv})',
                     f'emit("rb{k}", rb{k}.fit_transform(X))',
                     f'emit("rbi{k}", rb{k}.inverse_transform(X))',
                     f'emit_optional{"32" if dt == np.float32 else ""}("rbc{k}", rb{k}.center_)',
                     f'emit_optional("rbs{k}", rb{k}.scale_)']
            ref = P.RobustScaler(with_centering=wc, with_scaling=wsc, quantile_range=qr, unit_variance=uv).fit(X)
            expect(tag, f'rb{k}', ref.transform(X), dtype=dt)
            expect(tag, f'rbi{k}', ref.inverse_transform(X), dtype=dt)
            expect(tag, f'rbc{k}', ref.center_, 'optional')
            expect(tag, f'rbs{k}', ref.scale_, 'optional')
        d.case(tag, '\n'.join(body))

        # functions on both axes and on vectors
        tag = f'fn_{key}_{dt.__name__}'
        body = [f'var X = {d.input(X)}', f'var v = {d.input(np.ascontiguousarray(X[:, 0]))}']
        for axis in (0, 1):
            for wm, ws in ((True, True), (False, True), (True, False)):
                body.append(f'emit("sc{axis}{int(wm)}{int(ws)}", scale(X, axis={axis}, with_mean={wm}, with_std={ws}))')
                expect(tag, f'sc{axis}{int(wm)}{int(ws)}', P.scale(X, axis=axis, with_mean=wm, with_std=ws), dtype=dt)
            body += [f'emit("mm{axis}", minmax_scale(X, (-1.0, 1.0), axis={axis}))',
                     f'emit("ma{axis}", maxabs_scale(X, axis={axis}))',
                     f'emit("rb{axis}", robust_scale(X, axis={axis}, unit_variance=True))',
                     f'emit("vmm{axis}", minmax_scale(v, axis={axis}))',
                     f'emit("vma{axis}", maxabs_scale(v, axis={axis}))',
                     f'emit("vrb{axis}", robust_scale(v, axis={axis}))']
            expect(tag, f'mm{axis}', P.minmax_scale(X, (-1.0, 1.0), axis=axis), dtype=dt)
            expect(tag, f'ma{axis}', P.maxabs_scale(X, axis=axis), dtype=dt)
            expect(tag, f'rb{axis}', P.robust_scale(X, axis=axis, unit_variance=True), dtype=dt)
            v = np.ascontiguousarray(X[:, 0])
            expect(tag, f'vmm{axis}', P.minmax_scale(v, axis=axis), dtype=dt)
            expect(tag, f'vma{axis}', P.maxabs_scale(v, axis=axis), dtype=dt)
            expect(tag, f'vrb{axis}', P.robust_scale(v, axis=axis), dtype=dt)
        body.append('emit("vsc", scale(v))')
        expect(tag, 'vsc', P.scale(np.ascontiguousarray(X[:, 0])), dtype=dt)
        d.case(tag, '\n'.join(body))

# Errors and lifecycle edge cases.
err = {
    'inf': ('var s = StandardScaler()\ns.fit({inf})', ValueError),
    'quantile': ('var r = RobustScaler(quantile_range=(80.0, 20.0))\nr.fit({ok})', ValueError),
    'range': ('var m = MinMaxScaler((2.0, 1.0))\nm.fit({ok})', ValueError),
    'features': ('var s = StandardScaler()\ns.fit({ok})\n_ = s.transform({wide})', ValueError),
    'unfitted': ('var s = MaxAbsScaler()\n_ = s.transform({ok})', None),
    'weights_zero': ('var s = StandardScaler()\ns.fit({ok}, sample_weight={zw})', ValueError),
    'scale_axis': ('_ = scale({vec}, axis=1)', ValueError),
}
inputs = dict(inf=d.input(np.array([[1.0, np.inf]])), ok=d.input(np.ones((3, 2))), wide=d.input(np.ones((3, 3))),
              zw=d.input(np.zeros(3)), vec=d.input(np.ones(3)))
for name, (code, _) in err.items():
    d.case(f'err_{name}', code.format(**inputs))

out = d.run(imports='from moxie.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, scale, minmax_scale, maxabs_scale, robust_scale')

for case, name, value, kind, dtype in checks:
    got = out.get(case, {})
    label = f'{case}.{name}'
    if 'error' in got:
        d.close(label, got['error'], 0)
        continue
    if kind == 'count':
        arr, scalar, integer = emit_count_expected(value, dtype)
        d.close(label, got.get(name, 'missing'), arr, **STAT)
        d.equal(label + '.scalar', got.get(name + '_scalar'), scalar)
        d.equal(label + '.integer', got.get(name + '_integer'), integer)
    elif kind == 'optional':
        if value is None:
            d.equal(label, got.get(name + '_none'), 1)
        else:
            d.close(label, got.get(name, 'missing'), value, dtype=value.dtype, **STAT)
    elif kind == 'weighted':
        dt, extra = dtype
        tol = dict(TOL[dt], atol=TOL[dt]['atol'] + extra)
        d.close(label, got.get(name, 'missing'), value, dtype=dt, **tol)
    elif kind == 'int':
        d.equal(label, got.get(name), int(value))
    elif kind == 'exact':
        d.close(label, got.get(name, 'missing'), value, dtype=dtype, rtol=0, atol=0)
    else:
        d.close(label, got.get(name, 'missing'), value, dtype=dtype, **TOL[dtype])
for name in err:
    needle = 'NotFittedError' if name == 'unfitted' else 'ValueError'
    d.raises(f'err_{name}', out[f'err_{name}'].get('error'), needle)
d.finish()
