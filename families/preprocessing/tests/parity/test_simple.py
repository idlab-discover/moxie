"""Parity: Normalizer, Binarizer, KernelCenterer, normalize, binarize, add_dummy_feature.

Tolerances: Float64 rtol 1e-12 and Float32 rtol 4e-6 (atol 1e-6 for centered
kernels, which cancel). Norm accumulation order differs from NumPy's
pairwise/SIMD reductions only in final-bit rounding.
"""
import numpy as np

from harness import Driver, oracle

sk = oracle()
from sklearn import preprocessing as P

rng = np.random.RandomState(0)
d = Driver('simple')
expected = {}
TOL = {np.float64: dict(rtol=1e-12, atol=1e-14), np.float32: dict(rtol=4e-6, atol=1e-6)}
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}

data = {
    'gauss': rng.randn(7, 4),
    'zeros_row': np.vstack([np.zeros((1, 3)), rng.randn(5, 3)]),
    'tiny': np.array([[1e-300, -2e-300], [3.0, 4.0]]),
    'wide': rng.randn(2, 9) * 100,
    'single': np.array([[5.0]]),
}
for dt in (np.float64, np.float32):
    for key, raw in data.items():
        X = raw.astype(dt)
        T = d.input(X)
        tag = f'{key}_{dt.__name__}'
        body = [f'var X = {T}']
        for norm in ('l1', 'l2', 'max'):
            for axis in (0, 1):
                body.append(f'emit("n_{norm}_{axis}", normalize(X, "{norm}", axis={axis}))')
                body.append(f'var r_{norm}_{axis} = normalize_with_norms(X, "{norm}", axis={axis})')
                body.append(f'emit("nn_{norm}_{axis}", r_{norm}_{axis}[1])')
                Xn, norms = P.normalize(X, norm=norm, axis=axis, return_norm=True)
                expected[(tag, f'n_{norm}_{axis}')] = Xn
                expected[(tag, f'nn_{norm}_{axis}')] = norms
            body.append(f'var est_{norm} = Normalizer[{MOJO[dt]}]("{norm}")')
            body.append(f'emit("N_{norm}", est_{norm}.fit_transform(X))')
            body.append(f'var inplace_{norm} = X.copy()')
            body.append(f'est_{norm}.transform_inplace(inplace_{norm})')
            body.append(f'emit("Ni_{norm}", inplace_{norm})')
            expected[(tag, f'N_{norm}')] = P.Normalizer(norm=norm).fit_transform(X)
            expected[(tag, f'Ni_{norm}')] = expected[(tag, f'N_{norm}')]
        for threshold in (0.0, 0.5, -1.0):
            name = str(threshold).replace('.', 'p').replace('-', 'm')
            body.append(f'emit("b_{name}", binarize(X, threshold={threshold}))')
            body.append(f'var B_{name} = Binarizer[{MOJO[dt]}](threshold={threshold})')
            body.append(f'emit("B_{name}", B_{name}.fit_transform(X))')
            expected[(tag, f'b_{name}')] = P.binarize(X, threshold=threshold)
            expected[(tag, f'B_{name}')] = P.Binarizer(threshold=threshold).fit_transform(X)
        body.append('emit("dummy", add_dummy_feature(X))')
        body.append('emit("dummy7", add_dummy_feature(X, 7.5))')
        expected[(tag, 'dummy')] = P.add_dummy_feature(X)
        expected[(tag, 'dummy7')] = P.add_dummy_feature(X, 7.5)
        d.case(tag, '\n'.join(body))

# KernelCenterer on linear kernels, fit square, transform rectangular.
for dt in (np.float64, np.float32):
    A = rng.randn(6, 3).astype(dt)
    B = rng.randn(4, 3).astype(dt)
    K = (A @ A.T).astype(dt)
    Kt = (B @ A.T).astype(dt)
    tag = f'kernel_{dt.__name__}'
    kc = P.KernelCenterer().fit(K)
    expected[(tag, 'rows')] = kc.K_fit_rows_
    expected[(tag, 'all')] = np.array([kc.K_fit_all_])
    expected[(tag, 'fit')] = kc.transform(K)
    expected[(tag, 'test')] = kc.transform(Kt)
    expected[(tag, 'names')] = list(kc.get_feature_names_out())
    d.case(tag, f'''var K = {d.input(K)}
var Kt = {d.input(Kt)}
var kc = KernelCenterer[{MOJO[dt]}]()
kc.fit(K)
emit("rows", kc.K_fit_rows_)
var all = zeros[{MOJO[dt]}](Shape(1))
all.unsafe_set(0, kc.K_fit_all_)
emit("all", all)
emit("fit", kc.transform(K))
emit("test", kc.transform(Kt))
var inplace = Kt.copy()
kc.transform_inplace(inplace)
emit("inplace", inplace)
emit_names("names", kc.get_feature_names_out())''')
    expected[(tag, 'inplace')] = expected[(tag, 'test')]

# Error behavior: NaN/inf rejection, bad norm, non-square kernel.
nan_X = np.array([[1.0, np.nan]])
d.case('errors', f'''var X = {d.input(nan_X)}
try:
    _ = normalize(X)
    print("I nan_normalize 0")
except e:
    emit_error("nan_normalize", e)
try:
    _ = normalize({d.input(np.ones((2, 2)))}, "l3")
    print("I bad_norm 0")
except e:
    emit_error("bad_norm", e)
try:
    var kc = KernelCenterer()
    kc.fit({d.input(np.ones((2, 3)))})
    print("I nonsquare 0")
except e:
    emit_error("nonsquare", e)
try:
    var kc = KernelCenterer()
    _ = kc.transform({d.input(np.ones((2, 2)))})
    print("I unfitted 0")
except e:
    emit_error("unfitted", e)''')

out = d.run(imports='from moxie.preprocessing import Normalizer, Binarizer, KernelCenterer, normalize, binarize, add_dummy_feature\n'
                    'from moxie.preprocessing._simple import normalize_with_norms')
for (tag, name), value in expected.items():
    got = out.get(tag, {})
    if 'error' in got:
        d.close(f'{tag}.{name}', got['error'], value)
        continue
    if isinstance(value, list):
        d.equal(f'{tag}.{name}', got.get(name), value)
        continue
    dt = np.float32 if 'float32' in tag and name not in ('dummy', 'dummy7') else np.float64
    d.close(f'{tag}.{name}', got.get(name, 'missing'), value, dtype=value.dtype, **TOL[dt])
for name, needle in [('nan_normalize', 'NaN'), ('bad_norm', "'norm' parameter"), ('nonsquare', 'square'), ('unfitted', 'NotFittedError')]:
    d.raises(f'errors.{name}', out['errors'].get(name), needle)
d.finish()
